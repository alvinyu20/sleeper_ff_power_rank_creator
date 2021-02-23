import requests
import json
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import statistics
import matplotlib.pyplot as plt

#USER DEFINED VARIABLES
league_id = 600029829760348160
league_id_str = str(league_id)
season = 2020
current_week = 12
reg_weeks = 13
ros = [5,9,6,3,10,2,1,8,4,7]

#OTHER VARIABLES
base_url = 'https://api.sleeper.app/v1/league/'

#Query data from database
league_info = requests.get(base_url + league_id_str).json() #obtain all league information in dictionary
users = requests.get(base_url + league_id_str + '/users').json()
rosters = requests.get(base_url + league_id_str + '/rosters').json()

#Extract additional variables from league info
num_teams = league_info['total_rosters']
roster_positions = league_info['roster_positions']

#Extract additional variables from Users info
user_ids = [users[i]['user_id'] for i in range(num_teams)]
team_names = [users[i]['display_name'] for i in range(num_teams)]

#Extract additional variables from Rosters info
owner_ids = [rosters[i]['owner_id'] for i in range(num_teams)]
roster_ids = [rosters[i]['roster_id'] for i in range(num_teams)]
team_wins = [rosters[i]['settings']['wins'] for i in range(num_teams)]
team_losses = [rosters[i]['settings']['losses'] for i in range(num_teams)]
team_pf = [rosters[i]['settings']['fpts'] + (rosters[i]['settings']['fpts_decimal'] / 100) for i in range(num_teams)]
team_pa = [rosters[i]['settings']['fpts_against'] + (rosters[i]['settings']['fpts_against_decimal'] / 100) for i in range(num_teams)]

# Player database querying, only need to do every once in a while to update database
# player_database = requests.get('https://api.sleeper.app/v1/players/nfl').json()
# with open('player_database.txt','w') as data_file:
# json.dump(player_database,data_file)

with open('player_database.txt') as file:
    player_database = json.load(file)

'''
Create tables to store basic user info in
ref: Reference dataframe with user_id, username and roster_id
'''
user_ref = pd.DataFrame(list(zip(user_ids, team_names)), columns=['user_ids', 'user_name'])
roster_ref = pd.DataFrame(list(zip(owner_ids, roster_ids)), columns=['user_ids', 'roster_ids'])
ref = pd.merge(user_ref, roster_ref, on='user_ids')
ref = ref.sort_values(by=['roster_ids']).reset_index() #sort in order of roster ID

'''
Load rosters of every week into matrix
matchup_rosters: 4D list of every team's list of roster every week
weekly_points: 3D list of every team's weekly points
total_win_loss: 10x2 Dataframe showing win and losses of teams if they played everyone every week
'''
matchup_rosters = [[0 for x in range(num_teams)] for y in range(current_week)]
matchup_id = [[0 for x in range(num_teams)] for y in range(current_week)]
weekly_points = [[0 for x in range(num_teams)] for y in range(current_week)]
weekly_points_rank = [[0 for x in range(num_teams)] for y in range(current_week)]
total_win_loss = pd.DataFrame(0, index=ref['user_name'], columns=['W', 'L'])
for week in range(current_week):
    matchups = requests.get(base_url + league_id_str + '/matchups/' + str(week + 1)).json()
    for team in range(num_teams):
        matchup_rosters[week][team] = matchups[team]['starters']
        weekly_points[week][team] = matchups[team]['points']
    # calculate what rank each team is in terms of weekly points
    weekly_points_rank[week] = rankdata(weekly_points[week])
    total_win_loss['W'] += weekly_points_rank[week] - 1
    total_win_loss['L'] += num_teams - weekly_points_rank[week]

'''
Obtain full matchup schedule
schedule: all matchups for entire season mapped with integer
'''
schedule = pd.DataFrame(0, index=ref['user_name'], columns=range(1, reg_weeks + 1))
for week in range(reg_weeks):
    matchups = requests.get(base_url + league_id_str + '/matchups/' + str(week + 1)).json()
    for team in range(num_teams):
        schedule.iloc[team, week] = matchups[team]['matchup_id']

# simulate all remaining weeks
def montecarlo(iterations):
    '''
    Description: 
    Simulate remaining games using monte carlo simulation. This will be performed "iterations" number of times to createa playoff odds estimator

    Parameters:
    iterations  int: how many iterations of simulations to perform
    
    Return:
    playoff_odds    DataFrame: 10x10 table showing predicted playoff odds from Monte Carlo
    '''
    remaining_weeks = reg_weeks - current_week
    standings_count = pd.DataFrame(0, columns=range(1, num_teams + 1), index=ref['user_name'])
    playoff_odds = pd.DataFrame(0, columns=range(1, num_teams + 1), index=ref['user_name'])
    for iteration in range(iterations):
        standings_mc = pd.DataFrame(list(zip(team_wins, team_losses, team_pf)), columns=['W', 'L', 'PF'],
                                    index=range(num_teams))
        for week in range(current_week, reg_weeks):
            for matchup in range(int(num_teams/2)):
                standings_mc = simulate_matchup(week,matchup,standings_mc)
        # create win% for ranking
        standings_mc['win%'] = standings_mc['W'] / (standings_mc['W'] + standings_mc['L'])
        # create the ranking at the end, with tiebreaker being PF
        standings_mc = standings_mc.sort_values(by=['win%', 'PF'],ascending=False)
        standings_mc['Rank'] = np.arange(len(standings_mc)) + 1
        for i in range(num_teams):
            standings_count.loc[ref['user_name'][i], standings_mc['Rank'][i]] += 1
    playoff_odds = standings_count.div(iterations) * 100
    return playoff_odds


def simulate_winner(team_1, team_2):
    '''
    Description: 
    Simulate winner of a matchup to add to Monte Carlo standings
    Parameters:
    team_1  int: team id of first team
    team_2  int: team id of second team
    Return:
    team_1 or team_2: winning team
    team_1_score float
    team_2_score float
    '''
    # UPDATE stat to POINTS
    team_1_score = simulate_score(team_1) #simulate points for team 1
    team_2_score = simulate_score(team_2) #simulate points for team 2
    if team_1_score > team_2_score:
        return team_1, team_1_score, team_2_score
    elif team_2_score > team_1_score:
        return team_2, team_1_score, team_2_score

def simulate_score(team):
    '''
    Description: 
    Simulate score using sample from normal distribution using the team's mean and stdev
    Parameters:
    team int: team id to simulate score
    Return:
    simulated points
    '''
    return np.random.normal(team_mean[team], team_stdev[team], 1)

def simulate_matchup(week,i,standings_mc):
    '''
    Description: 
    Simulate score and winner of matchup. Next, update standings table for PF and wins and losses
    Parameters:
    week    int: week number to simluate for
    i       int: matchup id to simulate for (between 1 and 5)
    standings_mc: Monte carlo standings before simulation
    Return:
    standings_mc: Monte carlo standings after simulation
    '''
    # find team number in array
    team_1 = np.where(schedule[week + 1] == (i + 1))[0][0]
    team_2 = np.where(schedule[week + 1] == (i + 1))[0][1]
    winner, team_1_score, team_2_score = simulate_winner(team_1, team_2)
    standings_mc.loc[team_1, 'PF'] += team_1_score
    standings_mc.loc[team_2, 'PF'] += team_2_score
    # add wins and losses into standings table
    if winner == team_1:
        standings_mc.loc[team_1, 'W'] += 1
        standings_mc.loc[team_2, 'L'] += 1
    elif winner == team_2:
        standings_mc.loc[team_1, 'L'] += 1
        standings_mc.loc[team_2, 'W'] += 1
    return standings_mc

'''
Calculate mean and st dev for each team to use for monte carlo simuation
'''
team_mean, team_stdev = [], []
for i in range(num_teams):
    team_weekly_points_mc = [stat[i] for stat in weekly_points]
    team_weekly_points_mc = np.array(team_weekly_points_mc)
    team_weekly_points_mc = np.append(team_weekly_points_mc,np.random.normal((110 + (10 - ros[i]) * 2),18,10))
    #for j in range(10): #range(13-current_week):
        #team_weekly_points_mc.append((110 + (10 - ros[i]) * 2))
    team_mean.append(statistics.mean(team_weekly_points_mc))
    team_stdev.append(statistics.stdev(team_weekly_points_mc))

'''
Perform Monte Carlo Simulation
playoff_odds: Dataframe containing playoff odds and position odds for each team
'''
playoff_odds = montecarlo(2000)

'''
CREATE POWER RANKINGS
'''
power_rankings = pd.DataFrame(0, columns=['Team','Power Rank','Delta','Record','Ovr Record','Avg PPG','Playoff Odds','Bye Odds','ROS','PR Score'], index = range(num_teams))
power_rankings['Team'] = ref['user_name']
power_rankings['Record'] = [f"{str(i)}-{str(j)}" for i,j in zip(team_wins,team_losses)]
power_rankings['Ovr Record'] = [f"{str(int(i))}-{str(int(j))}" for i,j in zip(total_win_loss['W'],total_win_loss['L'])]
power_rankings['Avg PPG'] = [round(i / (current_week),1) for i in team_pf]
power_rankings['Playoff Odds'] = [f"{str(round(i,1))}%" for i in playoff_odds.iloc[:,0:6].sum(axis=1).reset_index()[0]]
power_rankings['Bye Odds'] = [f"{str(round(i,1))}%" for i in playoff_odds.iloc[:,0:2].sum(axis=1).reset_index()[0]]
power_rankings['ROS'] = ros

'''Create numerical rankings for all categories for calculating PR Score'''
record_rank = rankdata([-1 * i for i in team_wins])
ovr_record_rank = rankdata([-1 * i for i in total_win_loss['W']])
avg_ppg_rank = rankdata([-1 * i for i in power_rankings['Avg PPG']])
playoff_odds_rank = rankdata([-1 * i for i in power_rankings['Playoff Odds']])

#0record rank, 1ovr record rank, 2Avg PPG, 3playoff odss rank, 4ROS
weights = [3,4,5,3,4]
power_rankings['PR Score'] = round((record_rank*weights[0] + ovr_record_rank*weights[1] + avg_ppg_rank*weights[2] + playoff_odds_rank*weights[3] + power_rankings['ROS']*weights[4]) / sum(weights), 2)
power_rank = rankdata(power_rankings['PR Score'])
power_rankings['Power Rank'] = [f"#{str(int(i))}" for i in power_rank]
power_rankings['ROS'] = [f"#str(i)" for i in power_rankings['ROS']]

'''Calculate delta column based on change in rank from last week'''
power_rank_history = pd.read_csv(str(league_id) + '/power_rank_history.csv',header=0)
last_week = power_rank_history[str(current_week-1)]
for i in range(num_teams):
    if (power_rank[i] < last_week[i]):
        power_rankings.loc[i,'Delta'] = '^'
    elif (power_rank[i] > last_week[i]):
        power_rankings.loc[i,'Delta'] = 'v'
    else:
        power_rankings.loc[i,'Delta'] = '-'
power_rank_history[str(current_week)] = power_rank.tolist()

'''Sort by PR score'''
power_rankings = power_rankings.sort_values(by=['PR Score']).reset_index()

'''Export power rankings and playoff odds tables'''
html = power_rankings.to_html()
with open(str(league_id) + '/power_rankings.html','w') as data_file:
    data_file.write(html)
html = playoff_odds.to_html()
with open(str(league_id) + '/playoff_odds.html','w') as data_file:
    data_file.write(html)

power_rank_history.to_csv(str(league_id) + '/power_rank_history.csv',index=False)

#GRAPH 1:
ax1 = plt.subplot()
ax1.boxplot(weekly_points)
ax1.set_title('Team Scores per Fantasy Week')
plt.xlabel('Week')
plt.ylabel('Points')
plt.savefig(str(league_id) + '/scores_per_week.png')
plt.clf()

#GRAPH 2:
team_weekly_points = list(map(list, zip(*weekly_points)))
ax2 = plt.subplot()
ax2.boxplot(team_weekly_points)
ax2.set_title('Weekly Team Scores per Team')
ax2.set_xticklabels(playoff_odds.index, rotation=45)
plt.tight_layout()
plt.xlabel('Team')
plt.ylabel('Points')
plt.savefig(str(league_id) + '/team_scores_per_week.png')
plt.clf()