import streamlit as st 
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from pulp import LpMaximize,LpProblem,LpStatus,lpSum,LpVariable,LpInteger

st.set_page_config(
    page_title='FPL Analytics',
    layout="centered"
    )

@st.cache
def load_data():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()
    elements_df = pd.DataFrame(json['elements'])
    elements_types_df = pd.DataFrame(json['element_types'])
    teams_df = pd.DataFrame(json['teams'])
    slim_elements_df = elements_df[['id','second_name','team','element_type','selected_by_percent','now_cost','value_season','total_points','form','ict_index','dreamteam_count','in_dreamteam']]
    slim_elements_df['position'] = slim_elements_df.element_type.map(elements_types_df.set_index('id').singular_name)
    slim_elements_df['team_name'] = slim_elements_df.team.map(teams_df.set_index('id').name)
    slim_elements_df['value'] = slim_elements_df.value_season.astype(float)
    slim_elements_df['total_points'] = slim_elements_df.total_points.astype(float)
    slim_elements_df['ict_index'] = slim_elements_df.ict_index.astype(float)
    slim_elements_df['form'] = slim_elements_df.form.astype(float)
    slim_elements_df.sort_values('value',ascending=False,inplace=True)
    return slim_elements_df,elements_df

main_df,mapping_df = load_data()

def format_transfer_df(transfer_data,elements_df):
    transfer_history_df = pd.DataFrame(transfer_data)
    transfer_history_df['element_in_cost'] = transfer_history_df['element_in_cost']/10
    transfer_history_df['element_out_cost'] = transfer_history_df['element_out_cost']/10
    transfer_history_df['element_in_name'] = transfer_history_df.element_in.map(elements_df.set_index('id').second_name)
    transfer_history_df['element_out_name']= transfer_history_df.element_out.map(elements_df.set_index('id').second_name)
    transfer_history_df = transfer_history_df[['element_in_name','element_in_cost','element_out_name','element_out_cost','event']]
    transfer_history_df.columns = ['player in','player in cost','player out','player out cost','gameweek']
    transfer_history_df['gain or loss'] = transfer_history_df['player out cost'].astype(float) - transfer_history_df['player in cost'].astype(float)
    return transfer_history_df

def format_current_team_df(current_team_data,slim_elements_df,elements_df):
    team_df = pd.DataFrame(current_team_data)
    team_df['second_name'] = team_df.element.map(elements_df.set_index('id').second_name)
    team_df = team_df.merge(slim_elements_df,how='left',on='second_name')
    team_df = team_df[['element','team','second_name','position_y','selected_by_percent','now_cost','value','form','ict_index','dreamteam_count','in_dreamteam']]
    team_df['now_cost'] = team_df['now_cost'] / 10
    team_df.columns = ['id','team','name','position','selected by (%)','cost','value','form','ict score','dreamteam appearances','in dreamteam']
    team_df.sort_values('value',ascending=False,inplace=True)
    return team_df

def get_mean_variance(player_id):
    player_url = 'https://fantasy.premierleague.com/api/element-summary/{}/'.format(player_id)
    player_data = requests.get(player_url).json()
    player_df = pd.DataFrame(player_data['history'])
    mean = player_df['total_points'].mean()
    var = player_df['total_points'].std() ** 2
    return mean,var

@st.cache
def get_descriptive_statistics(player_dict):
    stats = {}
    mean = []
    for name in player_dict:
        individual_id = player_dict[name]
        player_mean,player_var = get_mean_variance(individual_id)
        stats[player_mean] = player_var
        mean.append(player_mean)
    mean.sort(reverse=True)
    mean = mean[:10]
    var = 0
    for m in mean:
        var += stats[m]
    return sum(mean),var ** (1/2)

@st.cache
def wildcard_suggestion(slim_elements_df,optimization_metric,current_team_value,weight):
    fpl_problem = LpProblem('FPL',LpMaximize)
    optimization_df = slim_elements_df[['second_name','team_name','team','total_points','position','now_cost','ict_index','form']]
    optimization_df = optimization_df.join(pd.get_dummies(optimization_df['position']))
    players = optimization_df['second_name']
    optimization_df['now_cost'] = optimization_df['now_cost']/10
    x = LpVariable.dict('x_ % s',players,lowBound=0,upBound=1,cat=LpInteger)

    metric_data = []

    point_arr = np.array(optimization_df['total_points'])
    point_norm = np.linalg.norm(point_arr)
    point_arr = point_arr/point_norm
    clean_player_points = dict(zip(optimization_df.second_name,np.array(optimization_df['total_points'])))
    player_points = dict(zip(optimization_df.second_name,point_arr))
    
    ict_arr = np.array(optimization_df['total_points'])
    ict_norm = np.linalg.norm(ict_arr)
    ict_arr = ict_arr/ict_norm
    clean_player_ict = dict(zip(optimization_df.second_name,np.array(optimization_df['ict_index'])))
    player_ict = dict(zip(optimization_df.second_name,ict_arr))

    form_arr = np.array(optimization_df['form'])
    form_norm = np.linalg.norm(form_arr)
    form_arr = form_arr/form_norm
    clean_player_form = dict(zip(optimization_df.second_name,np.array(optimization_df['form'])))
    player_form = dict(zip(optimization_df.second_name,ict_arr))

    if 'total_points' in optimization_metric:
        metric_data.append(player_points)
    
    if 'ict_index' in optimization_metric:
        metric_data.append(player_ict)

    if 'form' in optimization_metric:
        metric_data.append(player_form)

    final_data = {}
    for i in range(len(metric_data)):
        if i == 0:
            m = 'total_points'
        elif i == 1:
            m = 'ict_index'
        else:
            m = 'form'
        d = metric_data[i]
        for player in d:
            if player in final_data:
                final_data[player] += weight[m]*d[player]
            else:
                final_data[player] = weight[m]*d[player]

    fpl_problem += sum(final_data[i] * x[i] for i in players)

    position_names = ['Goalkeeper','Defender','Midfielder','Forward']
    position_constraints = [2,5,5,3]
    constraints = dict(zip(position_names,position_constraints))
    constraints['total_cost'] = float(current_team_value)
    constraints['team'] = 3

    player_cost = dict(zip(optimization_df.second_name, optimization_df.now_cost))
    player_position = dict(zip(optimization_df.second_name, optimization_df.position))
    player_gk = dict(zip(optimization_df.second_name, optimization_df.Goalkeeper))
    player_def = dict(zip(optimization_df.second_name, optimization_df.Defender))
    player_mid = dict(zip(optimization_df.second_name, optimization_df.Midfielder))
    player_fwd = dict(zip(optimization_df.second_name, optimization_df.Forward))

    fpl_problem += sum([player_cost[i] * x[i] for i in players]) <= float(constraints['total_cost'])
    fpl_problem += sum([player_gk[i] * x[i] for i in players]) == constraints['Goalkeeper']
    fpl_problem += sum([player_def[i] * x[i] for i in players]) == constraints['Defender']
    fpl_problem += sum([player_mid[i] * x[i] for i in players]) == constraints['Midfielder']
    fpl_problem += sum([player_fwd[i] * x[i] for i in players]) == constraints['Forward']

    for t in optimization_df.team.unique():
        optimization_df['team_'+str(t).lower()] = np.where(optimization_df.team == t, int(1),int(0))

    for t in optimization_df.team:
        player_team = dict(zip(optimization_df.second_name,optimization_df['team_'+str(t)]))
        fpl_problem += sum([player_team[i] * x[i] for i in players]) <= constraints['team']

    fpl_problem.solve()

    total_points = 0.
    total_cost = 0.
    optimal_squad = []
    for p in players:
        if x[p].value() != 0:
            total_points += clean_player_points[p]
            total_cost += player_cost[p]

            optimal_squad.append({
                'name': p,
                'position': player_position[p],
                'cost': player_cost[p],
                'points': clean_player_points[p],
                'ict_index':clean_player_ict[p],
                'form':clean_player_form[p]
            })

    solution_info = {
        'total_points': total_points,
        'total_cost': total_cost
    }
    optimal_squad_df = pd.DataFrame(optimal_squad)
    optimal_squad_df.sort_values('position',inplace=True)
    return optimal_squad_df,solution_info

st.title('FPL Team Insights')
st.subheader('Created by: Brandon Tan [Portfolio URL](https://brandontjd.github.io)')
st.write('Analyse your team - [How to find my team id?](https://www.reddit.com/r/FantasyPL/comments/4tki9s/fpl_id/) ')
st.write("While you find it, here's Magnus Carlsen's team!")
team_id = st.text_input('Enter your team id:',value='76862')

# Endpoints to call for manager data
manager_info_url = 'https://fantasy.premierleague.com/api/entry/{}/'.format(team_id)

manager_data = requests.get(manager_info_url).json()
try:
    if manager_data['detail'] == 'Not found.':
        st.write('Invalid team id found, please refer to the help above.')
        
except:
    current_game_week = manager_data['current_event']
    current_team_value = manager_data['last_deadline_value']/10
    overall_points = manager_data['summary_overall_points']
    manager_details_df = pd.DataFrame.from_dict({
        'id':manager_data['id'],
        'name':manager_data['player_first_name'] + ' ' + manager_data['player_last_name'],
        'region':manager_data['player_region_name'],
        'overall points':overall_points,
        'overall rank':manager_data['summary_overall_rank'],
        'current gameweek':current_game_week,
        'current team value': current_team_value

    },orient='index',columns=['Manager Data'])
    st.write(manager_details_df)

    #endpoints for team data
    transfer_url = 'https://fantasy.premierleague.com/api/entry/{}/transfers/'.format(team_id)
    picks_url = 'https://fantasy.premierleague.com/api/entry/{}/event/{}/picks/'.format(team_id,current_game_week)
    history_url = 'https://fantasy.premierleague.com/api/entry/{}/history/'.format(team_id)
    current_team_data = requests.get(picks_url).json()['picks']

    team_df = format_current_team_df(current_team_data,main_df,mapping_df)


#descriptive statistics 
descriptive_df = team_df[['id','name']]
player_dict = descriptive_df.set_index('name')
player_dict = player_dict['id'].to_dict()

mean,std = get_descriptive_statistics(player_dict)

st.header('Descriptive statistics')
st.write('This is based solely on current team selection')
st.subheader('Overall Team Expectation (adjusted with bench mean and var removed)')
st.write('Expected points per week:',mean)
st.write('Standard Deviation per week',std)
st.subheader('Statistics on individual player on team')
player = st.selectbox('Which player?',list(descriptive_df['name']))
player_id = player_dict[player]
player_url = 'https://fantasy.premierleague.com/api/element-summary/{}/'.format(player_id)
player_data = requests.get(player_url).json()
player_df = pd.DataFrame(player_data['history'])
player_df_home = player_df[player_df['was_home'] == True]
player_df_away = player_df[player_df['was_home'] == False]
player_home = player_df_home['total_points']
player_away = player_df_away['total_points']
hist_data = [player_home,player_away]
group_labels = ['home','away']
fig = ff.create_distplot(
    hist_data,group_labels
)
st.write('Home vs Away Distribution for {}'.format(player))
st.plotly_chart(fig,use_container_width=True)


##### ---- OPTIMIZER ---------
st.header('Wildcard optimization')
metrics = st.multiselect('Pick metrics to optimize on:', ['ict_index','total_points','form'],default=['total_points'])
st.write('You have chosen to optimize team based on: {}'.format(', '.join(metrics)))
st.write('Optimization will be done based on current team value of:',current_team_value)
weight = {}

st.write('Specify weights for optimization, with sum = 1 (this is optional)')
cols = st.beta_columns(len(metrics))
for i in range(len(cols)):
    weight[metrics[i]] = cols[i].number_input('{} weight:'.format(metrics[i]),value=1/len(metrics))

st.write('Total Weightage:',sum(weight.values()))
### ----- FUNC RUN -------
optimization_check = st.button('Run Optimization')
if optimization_check:
    if len(metrics) == 0:
        st.write('Please select at least 1 metric.')
    else:
        optimal_squad,solution_info = wildcard_suggestion(main_df,metrics,current_team_value,weight)
        st.subheader('Ideal Team')
        st.write(optimal_squad)
        list_of_different_players = list(optimal_squad[~optimal_squad['name'].isin(team_df['name'])]['name'])
        st.write('Players you are lacking:',', '.join(list_of_different_players))
        st.write('Number of transfers to create ideal team:',len(list_of_different_players))
        difference = float(solution_info['total_points'])-float(overall_points)
        st.write('Total Points of Optimal Team:',solution_info['total_points'], '(Difference in points:',difference,')')
        st.write('Total Cost of Optimal Team:',solution_info['total_cost'])

##### ---- GENERIC DATA ---------
st.header('Team Analysis')
st.subheader('Transfer History')
transfer_data = requests.get(transfer_url).json()
transfer_history_df = format_transfer_df(transfer_data,mapping_df)
st.write(transfer_history_df)

st.subheader('Current Team Data')
st.write(team_df)

st.header('Top Performers in each position that are not in your team')
top_100_df = main_df
recommendation_df = top_100_df[~top_100_df['second_name'].isin(team_df['name'])]
recommendation_df = recommendation_df[['team','second_name','position','selected_by_percent','now_cost','value','form','ict_index','dreamteam_count','in_dreamteam']]
recommendation_df.columns = ['team','name','position','selected by (%)','cost','value','form','ict score','dreamteam appearances','in dreamteam']
recommendation_df['cost'] = recommendation_df['cost']/10

st.write('Goalkeepers')
goalkeeper_df = recommendation_df[recommendation_df['position'] == 'Goalkeeper'].head(5)
st.write(goalkeeper_df)

st.write('Defenders')
defender_df = recommendation_df[recommendation_df['position'] == 'Defender'].head(5)
st.write(defender_df)

st.write('Midfielders')
midfielder_df = recommendation_df[recommendation_df['position'] == 'Midfielder'].head(5)
st.write(midfielder_df)

st.write('Forwards')
forward_df = recommendation_df[recommendation_df['position'] == 'Forward'].head(5)
st.write(forward_df)

st.title('General FPL Statistics')
st.write('This is intended to give you an overall feel for the entire league')

general_df = main_df.loc[main_df.value>0]
general_df['now_cost'] = general_df['now_cost']/10

pivot = general_df.pivot_table(index='position',values='value',aggfunc=np.mean).reset_index()
pivot.sort_values('value',ascending=False,inplace=True)
st.subheader('Average value by position')
st.write(pivot)

st.subheader('Average statistics by team')
team_pivot = general_df.pivot_table(index='team',values=['value','total_points','ict_index'],aggfunc=np.mean).reset_index()
team_pivot.sort_values('ict_index',ascending=False,inplace=True)
team_pivot.columns = ['team','average ict index','average points','average value']
st.write(team_pivot)

st.write('Total Points vs. Cost')
plotly_df = general_df[general_df['total_points'] > 50]
fig = px.scatter(plotly_df,x='now_cost',y='total_points',text='second_name',size='ict_index')
fig.update_traces(textposition='top center')
st.plotly_chart(fig)

st.write('Form vs. Points')
fig = px.scatter(plotly_df,x='form',y='total_points',text='second_name',size='ict_index')
fig.update_traces(textposition='top center')
st.plotly_chart(fig)

st.write('Selected by (%) vs. Total Points')
plotly_df = general_df[general_df['total_points'] > 50]
fig = px.scatter(plotly_df,x='selected_by_percent',y='total_points',text='second_name',size='ict_index')
fig.update_traces(textposition='top center')
st.plotly_chart(fig)

st.header('More Information')
st.write("I build this out of a desire to perform better in FPL. If you have any feature request, please direct them to my github repository, or you could drop me an email. My email can be found in my portfolio URL. Cheers.")

