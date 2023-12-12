#!/usr/bin/env python
# coding: utf-8


######################################## 
## Author : Sharath C R               ##
## SR No  : 13-19-01-19-52-22-1-21744 ##
## Date   : 10-May-2023               ##
########################################

# Import statements
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint

# Cleanup routine
def cleanup(df,uninterrupted_only=True,remove_errors=True,match_type="all",consider_extras=True):
    """
    Function to cleanup the data
    params:
    df -> Input Dataframe
    unin
    """
    # Get only innings 1 data.
    df_reduced = df[df.Innings == 1]
    
    # If only uninterrupted matches are to be considered. Default scenario.
    if uninterrupted_only:
        # Group match wise.
        df_reduced_grp = df_reduced.groupby(df_reduced.Match)
        # Get only uninterrupted matches data.
        valid = pd.DataFrame((df_reduced_grp['Wickets.in.Hand'].min() == 0) | (df_reduced_grp.Over.count() == 50))
        final=[]
        # Get all the indices for the uninterrupted matches.
        for i in [df_reduced[df_reduced.Match == i].index for i in valid[valid[0] == True].index]:
            final.extend(list(i))
        # Final clean df.
        df_reduced = df_reduced.loc[final, :]
    
    # Remove matchs that have reported errors in the data. Default scenario is all errors data points are removed.
    if remove_errors:
        df_reduced = df_reduced[df_reduced['Error.In.Data'] == 0]
    
    # By default, all matches will be considered.
    # If only Day-Night matches are to be considered.
    if match_type == "day_night":
        df_reduced = df_reduced[df_reduced['Day-night'] == 1]
    
    # If only Day matches are to be considered.
    if match_type == "day":
        df_reduced = df_reduced[df_reduced['Day-night'] == 0]
    
    # If extras in the match need to considered.
    if not consider_extras:
        # Update total runs coulmn using cumsum assuming data includes extra. 
        # Here assumption is per over data doesn't have extras included.
        df_reduced['Total.Runs'] = df_reduced.groupby('Match')['Runs'].cumsum()

        # Calculate innings total in each match
        matches = df_reduced['Match'].unique()
        innings_totals = [df_reduced.loc[df_reduced['Match'] == m, 'Total.Runs'].max() for m in matches]

        # Create a reduced dataframe with only the Match and Over columns.
        df_match_over = df_reduced[['Match', 'Over']].copy()
        df_match_over['Innings.Total.Runs'] = 0

        # Iterate over the rows of the reduced dataframe and add Innings.Total.Runs
        for i, row in df_match_over.iterrows():
            match_index = matches.tolist().index(row['Match'])
            df_match_over.at[i, 'Innings.Total.Runs'] = innings_totals[match_index]

        # Assign the innings total runs to the original dataframe.
        df_reduced['Innings.Total.Runs'] = df_match_over['Innings.Total.Runs']
        
        # Update the Run.Remaining using the updated columns.
        df_reduced['Runs.Remaining'] = df_reduced['Innings.Total.Runs'] - df_reduced['Total.Runs']
    
    return df_reduced

## Objective
def objective_fun(Z0, L, u):
    return Z0 * (1 - np.exp(-L*u/Z0))

## Function to initialize the parameters.
def initialize_z_w(data, w, init_type='mean'):
    data = data[data['Wickets.in.Hand'] == w]
    if init_type == 'mean':
        runs = data.groupby(['Match'])['Runs.Remaining'].mean()
    elif init_type == 'max':
        runs = data.groupby(['Match'])['Runs.Remaining'].max()
    else:
        print("Invalid initialization type")
        sys.exit(1)
    return np.mean(runs)


## Error function
def error(params,data,norm_err=True,print_debug=False):
    # Get all the 11 parameters
    Z0 = params[:10]
    L = params[10]
    
    # Initialize the errors
    err1 = err2 = 0
    
    # Iterate through all the wickets, essentially going through
    # all the data points in the data set.
    for i in range(1,11):
        # This is to capture the case where there are 50 overs to go as this will be
        # missed when we do (50 - u) to calculate the remaining overs.
        # i == 10 is needed because this condition can only happen at the beginning of a match
        if i == 10:
            y50 = objective_fun(Z0[9], L, 50)
            # 'Over' == 1 is needed because we are looking for the beginning of the match and
            # 'Innings.Total.Runs' is taken because the the final score with 50 overs and 10 wickets in hand.
            err2 += np.sum((y50 - np.array(data[(data['Over'] == 1)]['Innings.Total.Runs']))**2)
        # else: ( This will be always true )
        y = objective_fun(Z0[i-1], L, 50-df_reduced[df_reduced['Wickets.in.Hand'] == i]['Over'].values)
        err1 += np.sum((y - df_reduced[df_reduced['Wickets.in.Hand'] == i]['Runs.Remaining'].values)**2)
    # Now the normalized error is minimized if norm_err = True. This will be the default condition.
    if norm_err:
        err2 /= len(np.array(data[(data['Over'] == 1)]['Innings.Total.Runs']))
        err1 /= len(data)
    
    err = err1 + err2
    if print_debug:
        print("Error this iter: {:5f}".format(err))
    return err

# Report out the 11 Parameters:
def print_table(headers, data):
    # Calculate the width of each column
    widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
    
    print('=' * (np.sum(np.array(widths)+3) + len(headers) - 1))
    
    # Print the headers
    print(f'{" | ".join(header.upper().center(widths[i]+1) for i, header in enumerate(headers))}')

    # Print the separator line
    print('-' * (np.sum(np.array(widths)+3) + len(headers) - 1))

    # Print the rows
    for row in data:
        print(f'{" | ".join(str(row[i]).ljust(widths[i]+1) for i in range(len(headers)))}')
    
    print('=' * (np.sum(np.array(widths)+3) + len(headers) - 1))
    
if __name__ == '__main__':
    
    # Read the Data
    try:
        path_to_data = r"{}".format(os.path.join(os.path.abspath("../data"),"04_cricket_1999to2011.csv"))
        df = pd.read_csv(path_to_data)
    except Exception as e:
        print("[ERROR]: Couldn't read the data file. Make sure its in the correct path.\n")
        print("[INFO]: Looking the following path: {}".format(path_to_data))
        sys.exit(1)
    
    # Control variables for running different experiments
    uninterrupted_only = True 
    remove_errors = True
    match_type = 'all'
    consider_extras = True
    init_type = 'mean'
    norm_err=True
    print_debug=True
    
    # Cleanup the data
    df_reduced = cleanup(df, 
                         uninterrupted_only=uninterrupted_only, 
                         match_type=match_type, 
                         remove_errors=remove_errors, 
                         consider_extras=consider_extras)
    
    # Initialize the Parameters with respect to Data. Observed that this type of initialization
    # helps with convergence to lower error function value.
    params = []
    for i in range(1,11):
        params.append(initialize_z_w(df_reduced, i,
                                     init_type=init_type))
        #params.append(100)
        
    # Initialize L
    params.append(15)
    
    print("Parameters initialized to : ",params)
    print("\n")
    
    # Bounded and Constrained Optimization
    # -------+
    # Bounds |
    # -------+
    # 0 < (Z[0], Z[1], Z[2], Z[3], Z[4], Z[5], Z[6], Z[7], Z[8], Z[9], L) < inf
    #
    # ------------+
    # Constraints |
    # ------------+
    # 0 < Z[0] < Z[1] < Z[2] < Z[3] < Z[4] < Z[5] < Z[6] < Z[7] < Z[8] < Z[9] < inf
    # 0 < L < inf
    #
    # How this is checked? By taking difference and checking the bounds on the difference.
    # inf > Z[1] - Z[0] > 0

    bounds = tuple([(0,np.inf)]*11)

    def constraint(x):
        x = x[:10]
        return [x[i+1] - x[i] for i in range(len(x)-1)]

    # Define the inequality constraint object
    linear_constraint = LinearConstraint(np.identity(11), [0]*10 + [0], [np.inf]*10 + [np.inf])

    # Perform constrained and bounded Optimization. L-BFGS-B supports this as per scipy documentation.
    print("Performing Optimization..\n")
    opt = minimize(error, params, args=(df_reduced, norm_err, print_debug),method='L-BFGS-B',bounds=bounds,constraints=[linear_constraint, {'type': 'ineq', 'fun': constraint}])
    
    print("\n\nOptimal points found..\n\n")
    try:
        print("Z0's")
        print_table(['Z0({i})'.format(i=i) for i in range(1,11)], [list(np.round(opt.x[:10],2))])
        print("\n")
        print_table(['L'],[[np.round(opt.x[10],2)]])
    except:
        print("Priting in Table format failed")
        print("Z0's : ")
        print(opt.x[:10])
        print("\nL")
        print(opt.x[10])


    # Generate plots
    w = 11
    plt.figure()
    overs = np.linspace(0, 50, num=500)
    for i in np.flip(opt.x[0:10]):
        w -= 1
        plt.plot(overs,objective_fun(i,opt.x[10],overs),label=w)
    plt.title("Run Production Functions")
    plt.xlabel('Remaining Overs')
    plt.ylabel('Average Runs Obtainable')
    plt.xlim((0, 50))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.grid()
    #plt.legend(loc='upper right', bbox_to_anchor=(1.27,1.02),title='Wickets In Hand')
    leg = plt.legend(title='Wickets In Hand',ncols=2)
    leg.get_title().set_fontsize('11')
    plt.savefig('run_production_1.jpg',dpi=900)
    plt.show(block=False)
    

    w = 11
    plt.figure()
    for i in np.flip(opt.x[:10]):
        w -= 1
        plt.plot(overs, objective_fun(i,opt.x[10],overs)/objective_fun(opt.x[9],opt.x[10],50)*100, label=w)
    plt.title("Resources Remaining (%)")
    plt.xlabel('Remaining Overs')
    plt.ylabel('%')
    plt.xlim((0, 50))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.grid()
    leg = plt.legend(title='Wickets In Hand',ncols=2)
    leg.get_title().set_fontsize('11')
    plt.savefig('resources_1.jpg',dpi=900)
    plt.show()


