#
# mint2tiller.py
#
# Reformat a Mint export for Tiller, and attempt to deduplicate overlapped entries.
#

import pandas as pd
import argparse
import re
import os
import pprint
import numpy as np
import itertools
import string

TILLER_COLS = [ "Date", "Description", "Category", "Amount", "Account", "Account #", "Institution", "Month", "Week", "Transaction ID", "Check Number", "Full Description", "Date Added" ]
SPLICE_OVERLAP_DAYS = 7

class MintFormatError( ValueError ):
    """
    Indicates that the Mint data received does not fit our expectations.
    """
    pass

def read_from_tiller( csv_path ):
    """
    Read a tiller export. This is what you downloaded from Google Drive after Tiller's
    initial data pull.

    Actually, there's nothing special that this function does right now. It's defined
    to wrap DataFrame.read_csv() to provide a stable API.
    """
    df = pd.read_csv( csv_path )
    df[ "Date" ] = pd.to_datetime( df[ "Date" ] )
    df[ "Amount" ] = df[ "Amount" ].str.replace( r'[^-+\d.]', '' ).astype( float )
    return df

def read_from_mint( csv_path, category_map=None ):
    """
    Read in a Mint export, reformat it to Tiller Money's format, and return the DataFrame.

    category_map: Dict or collections.defaultdict that keys on a Mint category name and whose
                  values indicate Tiller Foundation Template category names.
    """
    #
    # Read data with proper column types
    #
    df = pd.read_csv( csv_path, parse_dates=True, infer_datetime_format=True )
    df[ "Date" ] = pd.to_datetime( df[ "Date" ] )
    df[ "Amount" ] = df[ "Amount" ].astype( float )

    # Remove unprintable characters and streams of random carriage returns (wtf Mint?)
    printable_chars = set( string.printable ) - set( [ '\r' ] )
    for colname in [ "Description", "Original Description" ]:
        df[ colname ] = df[ colname ].apply( lambda x: "".join( [ c for c in x if c in printable_chars ] ) )

    #
    # Lint the file so we don't do anything stupid later
    #

    # All "Transaction Type" must be either 'debit' or 'credit'
    if len( set( df[ "Transaction Type" ].unique() ) - set( [ "debit", "credit" ] ) ) > 0:
        raise MintFormatError( "Unknown or missing transaction type present in file." )

    # All "Amount" must be at least zero
    if not all( df[ "Amount" ] >= 0 ):
        raise MintFormatError( "Expected all amounts to be non-negative." )

    #
    # Change the format to match Tiller's template
    #

    # Remap categories
    if category_map is None:
        df[ "Category" ] = None
    else:
        df[ "Category" ] = df[ "Category" ].map( category_map )

    # Tiller's "Account" maps to Mint's "Account Name"
    # Can't determine "Account #" here, set blank
    # Can't determine institution here, set blank
    df[ "Account" ] = df[ "Account Name" ]
    df[ "Account #" ] = None
    df[ "Institution" ] = None

    # Guess check number with regex
    df[ "Check Number" ] = df[ "Description" ].str.extract( r"Check (?P<check_number>\d+)", expand=False, flags=re.IGNORECASE )

    # Fill in Start of Week and Month columns
    df[ "Week" ] = df[ "Date" ].values.astype( 'datetime64[W]' )
    df[ "Month" ] = df[ "Date" ].values.astype( 'datetime64[M]' )

    # Tiller's "Full Description" is Mint's "Original Description"
    # The pretty version is just called "Description" in both -- no work there.
    df[ "Full Description" ] = df[ "Original Description" ]

    # Invert the sign of all debit transactions
    #assert( any( df[ "Transaction Type" ] == "debit" ) )
    df[ "Amount" ][ df[ "Transaction Type" ] == "debit" ] *= -1

    # It's being added to Tiller 'today', and Mint doesn't offer anything
    # better to place in this column
    df[ "Date Added" ] = pd.to_datetime( "today" )

    # There won't be any transaction IDs
    df[ "Transaction ID" ] = None

    # Reorder/Drop columns, then return the dataframe
    df = df[ TILLER_COLS ]

    return df

def merge_dataframes( mint_df, tiller_df, verbose=False ):
    """
    "Intelligently" determine when/where to splice the data from each account in both sources.
    This is slightly tricky for a few reasons:

    1. When tiller pulls transaction histories for an account, there are no clear rules on how
       far back it will be able to pull from. Some accounts may be 90 days, and others less.
       This function determines the splice date on an account-by-account basis.
    2. There's no key on which to merge rows between Mint and Tiller. The description fields
       may often match, but not always. The dates do not always align either, and can be off
       by a few days. To work around this, a window is determined around each account's splice
       date, and transactions are matched between Mint and Tiller heuristically. In each case,
       the Tiller portion of the match is added to the output. Any unmatched transactions are
       added to the output.
    """
    merged_df = pd.DataFrame( columns=TILLER_COLS )
    mint_accounts = set( mint_df[ "Account" ].unique() )
    tiller_accounts = set( tiller_df[ "Account" ].unique() )

    mint_exclusive_accounts = mint_accounts - tiller_accounts
    tiller_exclusive_accounts = tiller_accounts - mint_accounts
    overlapped_accounts = tiller_accounts & mint_accounts

    if verbose:
        print( "Accounts only in Mint:" )
        pprint.pprint( mint_exclusive_accounts )
        print( "Accounts only in Tiller:" )
        pprint.pprint( tiller_exclusive_accounts )
        print( "Accounts in both: " )
        pprint.pprint( overlapped_accounts )

    # Any accounts in exclusively mint or tiller can go directly
    # into the output df
    mint_exclusive_transactions = mint_df[ mint_df[ "Account" ].isin( mint_exclusive_accounts ) ]
    tiller_exclusive_transactions = tiller_df[ tiller_df[ "Account" ].isin( tiller_exclusive_accounts ) ]
    merged_df = pd.concat( ( merged_df, mint_exclusive_transactions, tiller_exclusive_transactions ), sort=False )

    # Now, we need to merge things. Do so on an account by account basis
    for account_name in overlapped_accounts:

        if verbose:
            print( "Processing account {}.".format( account_name ) )

        # Get the transactions pertinent to this account
        these_mint_transactions = mint_df[ mint_df[ "Account" ] == account_name ]
        these_tiller_transactions = tiller_df[ tiller_df[ "Account" ] == account_name ]

        # Any transactions in mint more than SPLICE_OVERLAP_DAYS older than the oldest Tiller transaction
        # are almost certainly not going to have a duplicate in the tiller data -- fast forward it to
        # the results dataframe.
        mint_bypass_date = these_tiller_transactions[ "Date" ].min() - pd.to_timedelta( SPLICE_OVERLAP_DAYS, unit="d" )
        mint_bypassed_transactions = these_mint_transactions[ these_mint_transactions[ "Date" ] <= mint_bypass_date ]
        merged_df = pd.concat( ( merged_df,  mint_bypassed_transactions ), sort=False )
        if verbose:
            print( "Passing {} of {} Mint transactions which are sufficiently older than the oldest Tiller transaction".format(
                len( mint_bypassed_transactions ),
                len( these_mint_transactions ) ) )
        these_mint_transactions = these_mint_transactions[ these_mint_transactions[ "Date" ] > mint_bypass_date ]

        # Now, the two these_*_transactions dataframes should contain mostly duplicated entries, plus a few in the
        # mint dataframe which are not present in Tiller (due to our SPLCE_OVERLAP_DAYS ). We expect every transaction
        # in tiller to have a transaction in Mint, but not vice-versa -- the Tiller data should subset the Mint data.
        # When it does not, we will print an error message, and simply append the mismatched transactions.

        # First, we can pick out any transactions whose amounts occur exclusively in one dataframe or the other, but
        # not both. Amount is the only field that we can count on to match between sources.
        these_mint_exclusive_amounts = set( these_mint_transactions[ "Amount" ] ) - set( these_tiller_transactions[ "Amount" ] )
        these_tiller_exclusive_amounts = set( these_tiller_transactions[ "Amount" ] ) - set( these_mint_transactions[ "Amount" ] )

        # ...for mint...
        bypass_mask = these_mint_transactions[ "Amount" ].isin( these_mint_exclusive_amounts )
        mint_bypassed_transactions = these_mint_transactions[ bypass_mask ]
        merged_df = pd.concat( ( merged_df,  mint_bypassed_transactions ), sort=False )
        # TODO Issue warning if date is not near cutoff date
        if verbose:
            print( "Passing {} of {} Mint transactions which have no matching dollar amount in Tiller".format(
                len( mint_bypassed_transactions ),
                len( these_mint_transactions ) ) )
        these_mint_transactions = these_mint_transactions[ ~bypass_mask ]

        # ...for Tiller...
        bypass_mask = these_tiller_transactions[ "Amount" ].isin( these_tiller_exclusive_amounts )
        tiller_bypassed_transactions = these_tiller_transactions[ bypass_mask ]
        merged_df = pd.concat( ( merged_df,  tiller_bypassed_transactions ), sort=False )
        # TODO Issue warning if date is not near cutoff date
        if verbose:
            print( "Passing {} of {} Tiller transactions which have no matching dollar amount in Mint".format(
                len( tiller_bypassed_transactions ),
                len( these_tiller_transactions ) ) )
        these_tiller_transactions = these_tiller_transactions[ ~bypass_mask ]

        # Now, the hard part. Find an optimal matching for all transactions of
        # the same amount. We basically want to minimize the total error in terms
        # of Date.
        #
        # TODO Optimization Opportunity: This could be vectorized with Numpy. But, considering the short date window,
        # probably not worth it.
        assert( set( these_mint_transactions[ "Amount" ] ) == set( these_tiller_transactions[ "Amount" ] ) )
        for amount in set( these_mint_transactions[ "Amount" ] ):
            t = these_tiller_transactions[ these_tiller_transactions["Amount"] == amount ]
            m = these_mint_transactions[ these_mint_transactions["Amount"] == amount ]

            #
            # Find the cheapest alignment. There's a name for this algorithm but I don't
            # remember it. It's brute force and N^2 so, yeah, banking on N being smallish.
            # There are smarter ways to do this, but for small N, it's faster for me to
            # just reinvent the wheel.
            #
            t_dates = list( t[ "Date" ] )
            m_dates = list( m[ "Date" ] )

            # Pad the shorter of the two with Nones
            if len( t_dates ) < len( m_dates ):
                t_dates.extend( [None] * ( len( m_dates ) - len( t_dates ) ) )
            else:
                m_dates.extend( [None] * ( len( t_dates ) - len( m_dates ) ) )
            dims = len( m_dates )

            # Find the cost matrix in terms of days of error between Date values
            cost_matrix = np.ones( ( dims, dims ) ) * np.nan
            for i, t_date in enumerate( t_dates ):
                for j, m_date in enumerate( m_dates ):
                    cost_matrix[ i, j ] = abs( ( t_date - m_date ).days ) if ( t_date and m_date ) else 999

            # Take the cheapest match set along the smaller axis
            # This is also brute force 🤷
            best_alignment = None
            lowest_cost = None
            for alignment in itertools.permutations( list( range( dims ) ), dims ):
                costs = [ cost_matrix[ i, j ] for i, j in enumerate( alignment ) ]
                cost = sum( costs )
                if best_alignment is None or lowest_cost > cost:
                    best_alignment = alignment
                    lowest_cost = cost

            # Comment on how we ended up deduplicating things
            for i in range( dims ):
                j = best_alignment[ i ]
                tiller_row = t.iloc[ i ] if t_dates[ i ] is not None else None
                mint_row = m.iloc[ j ] if m_dates[ j ] is not None else None
                if tiller_row is not None and not mint_row is not None:
                    if verbose:
                        print( "Found no match for the following Tiller transaction -- keeping it.\n{}\n".format( tiller_row ) )
                    merged_df = merged_df.append( tiller_row )
                elif mint_row is not None and not tiller_row is not None:
                    if verbose:
                        print( "Found no match for the following Mint transaction -- keeping it.\n{}\n".format( mint_row ) )
                    merged_df = merged_df.append( mint_row )
                else:
                    if verbose:
                        print( "Found a duplicate transaction -- keeping the Tiller version.\n{}\n".format( pd.DataFrame( [ tiller_row, mint_row ] ) ) )
                    merged_df = merged_df.append( tiller_row )

        # Done!
        if verbose:
            print( "Done!" )

    # Sort by date
    merged_df.sort_values( "Date", inplace=True, ascending=False )

    return merged_df

def write_to_tiller( df, out_path ):
    """
    Write a dataframe in the Tiller format (mostly just special date format settings)
    """
    # TODO differnt date format for some cols
    # TODO format currency
    for colname in [ "Date", "Date Added", "Month", "Week" ]:
        df[ colname ] = pd.to_datetime( df[ colname ] )
    #df.to_csv( out_path, date_format="%-d/%-m/%Y", index=False )
    df.to_csv( out_path, index=False )

def mint2tiller( mint_csv_path, tiller_initial_csv_path, output_path, category_map=None, verbose=False ):
    """
    Full reformat+merge flow.

    1. Reads and reformats mint_csv_path using read_from_mint() and category_map
    2. Merges it with tiller_initial_csv_path, removing duplicates, via merge_dataframes().
    3. Writes the result to output_path with write_to_tiller()
    """
    mint_df = read_from_mint( mint_csv_path, category_map=category_map )
    tiller_df = read_from_tiller( tiller_initial_csv_path )
    merged_df = merge_dataframes( mint_df, tiller_df, verbose=verbose )
    write_to_tiller( merged_df, os.path.expanduser( output_path ) )
    return merged_df

if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser( description="Merge a Mint transactions export into a newly configured Tiller Money export.." )
    parser.add_argument( "--mint_csv_path", type=str, help="(Input) Path to your Mint transactions CSV." )
    parser.add_argument( "--tiller_initial_csv_path", type=str, help="(Input) Path to your Tiller Money transactions CSV (after initial data pull & download from Google Drive)." )
    parser.add_argument( "--output_path", type=str, help="(Output) Where to place the merged CSV, which will be uploaded back to Google Drive." )
    parser.add_argument( "--verbose", action="store_true", help="Whether to print useful information." )
    cli_kwargs = { k: v for k, v in vars( parser.parse_args() ).items() if v is not None }

    # Do it
    mint2tiller( **cli_kwargs )
