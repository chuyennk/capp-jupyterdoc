import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import seaborn as sns

from mylibs.table_display import (docTable, formatMultiIndexTable)
# from mylibs.configs import capp_config, scoreband_config, factor_config, backend_test_thresholds

def calcAR(total: np.array, bad: np.array, sortby:np.array=None, ascending=False):
    """
    To compute AR & AUC using sklearn roc_auc_score. 
    To compute by bucketing, need to use sample_weight by duplicating good & bad record 
    by number of good & bad accordingly
    
    Note: df should not contain the Total fields
    """
    if np.nansum(bad) == 0:
        return 0, 0
    else:
        dfAR = None
        if type(sortby) == 'NoneType':
            dfAR = pd.DataFrame({'total': total, 'bad': bad})
        else:
            dfAR = pd.DataFrame({'total': total, 'bad': bad, 'sortby': sortby})
            dfAR.sort_values(by='sortby', ascending=ascending, inplace=True)

        dfAR = dfAR[dfAR['total'].notnull()]
        
        # print(dfAR)

        y_pred = np.append(np.arange(dfAR.shape[0]),np.arange(dfAR.shape[0]))
        y_actual = np.append(np.zeros(dfAR.shape[0]), np.ones(dfAR.shape[0]))
        sample_weight = np.append(dfAR['total'] - dfAR['bad'], dfAR['bad'])
        
        # the practise is to show low score, high bad rate on top. Hence, take 1 - AUC to ensure positive AUC & AR 
        AUC = (1 - roc_auc_score(y_actual, y_pred, sample_weight=sample_weight))
        
        AR = (AUC - 0.5)/0.5
    
    return AR, AUC

def calcRiskRankingTable(df, configs):
    """
    compute distribution and bad rate by scoreband
    return the result in a dataframe and a formatted html string for display
    """
    capp_config = configs.get('capp_config')
    scoreband_config = configs.get('scoreband_config')

    df = df.copy()
    df['scoreband'] = pd.cut(df[capp_config['SCORE']], scoreband_config['scoreband'])

    dfBackend = df.pivot_table(
        index = 'scoreband',
        columns = capp_config['COHORT_GROUP'],
        aggfunc={capp_config['BAD']: {'count','sum'}},
        observed = True
    )
    
    dfBackend.columns = dfBackend.columns.droplevel(0)
    dfBackend.rename(columns={'count':'Count', 'sum': capp_config['BAD_NAME']}, inplace=True)

    dfOutcome, styledOutcome = getRiskRankingOutcome(dfBackend, bucket=dfBackend.index.get_level_values(0).values, configs=configs)
    
    # begin styling the backend table
    for col in dfBackend.columns:
        if col[0] == 'Count':
            total = dfBackend[col].sum()
            dfBackend[('Distribution', col[1], col[2])] = dfBackend[col]/total
        if col[0] == capp_config['BAD_NAME']:
            dfBackend[(capp_config['BADRATE_NAME'], col[1], col[2])] = dfBackend[col]/dfBackend[('Count', col[1], col[2])]

    styledBackend = dfBackend.copy()
    styledBackend.columns.names = [None, None, None]
    styledBackend.index.name = None

    styledBackend.loc[('Total'),:] = styledBackend.sum(axis=0)
    badrates = (styledBackend.loc['Total', capp_config['BAD_NAME']] / styledBackend.loc['Total', 'Count']).values
    styledBackend.loc['Total',capp_config['BADRATE_NAME']] = badrates

    styledBackend = docTable(styledBackend,
                            show_index = True,
                            table_type = 'text-center',
                            caption = 'Distribution & Bad Rate by Scoreband',
                            numbers_format=formatMultiIndexTable(cols=styledBackend.columns, 
                                              firstIndexNames=['Count', capp_config['BAD_NAME'], 'Distribution', capp_config['BADRATE_NAME']],
                                              formats =['{:,.0f}', '{:,.0f}', '{:,.1%}','{:,.2%}']
                                             ))
    # end styling the backend table


    return dfOutcome, dfBackend, styledOutcome, styledBackend

def getRiskRankingOutcome(dfAR, bucket, isFactorLevel=False, configs=None):
    capp_config = configs.get('capp_config')
    backend_test_thresholds = configs.get('backend_test_thresholds')
    thresholds = backend_test_thresholds['model_ranking']
    if isFactorLevel:
        thresholds = backend_test_thresholds['factor_ranking']

    combined_thresholds = {}
    for threshold in thresholds.get('combined'):
        combined_thresholds[tuple(threshold.get('key'))] = threshold.get('value')

    cohorts = dfAR.loc[:, 'Count'].columns.get_level_values(1)
    # bucket = dfAR.index.astype(str).values  
    total = dfAR.loc[:, 'Count'].values
    bad = dfAR.loc[:, capp_config['BAD_NAME']].values

    dfOutcome = pd.DataFrame(columns = cohorts)
    for idx, cohort in enumerate(cohorts):
        AR, AUC = calcAR(total[:,idx], bad[:,idx], bucket, ascending=True)

        dfOutcome.loc['AR', cohort] = AR
        dfOutcome.loc['AUC', cohort] = AUC
        dfOutcome.loc[f"# of {capp_config.get('BAD_NAME')}", cohort] = np.nansum(bad[:,idx])
        
        if idx==0:
            baselineAR = AR
            outcome = pd.cut([AR], bins=thresholds['absolute'], labels=thresholds['absolute_outcome'], ordered=False)[0]
        else:
            strRel = pd.cut([AR - baselineAR], bins=thresholds['relative'], labels=thresholds['relative_outcome'], ordered=False)[0]
            strAbs = pd.cut([AR], bins=thresholds['absolute'], labels=thresholds['absolute_outcome'], ordered=False)[0]
            outcome = combined_thresholds[(strRel, strAbs)]
            
        dfOutcome.loc['Outcome', cohort] = outcome

    outcomeHighlight = {'GREEN': 'outcome-complied', 'YELLOW': 'outcome-moderate', 'RED': 'outcome-significant'}
    dfHighlight = dfOutcome.applymap(lambda x: outcomeHighlight.get(x, ''))

    sOutcome = docTable(dfOutcome, 
         show_index=True,
         caption='Risk Ranking Outcome',
         table_type = 'text-center',
         numbers_format='{:,.1%}', 
         highlight_cells=dfHighlight
        )

    # Override the format for # of bad
    sOutcome.format('{:,.0f}', subset=pd.IndexSlice['# of ' + capp_config['BAD_NAME'],:])

    return dfOutcome, sOutcome
        
def getRiskRankingChart(bucket, dist, badRate, title='Distribution & Bad Rate by Scoreband', configs=None):
    capp_config = configs.get('capp_config')

    plt.rcParams.update({'font.size': 8})
    fig, ax1 = plt.subplots(figsize=(4,3))

    ax2 = ax1.twinx()
    
    chart1 = sns.lineplot(x=bucket, y=badRate, color='red', linewidth=2, alpha=1, ax=ax1)
    chart2 = sns.barplot(x=bucket, y=dist, color='blue', alpha=1, ax=ax2)
    chart1.set_xticklabels(chart1.get_xticklabels(), rotation=45)
    chart2.set_xticklabels(chart2.get_xticklabels(), rotation=45)
    
    ax1.set_title(title)
    ax1.set(xlabel=None, ylabel=capp_config['BADRATE_NAME'])
    ax2.set(xlabel=None, ylabel="Count")

    ax1.get_yaxis().set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax2.get_yaxis().set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # bring ax1 to front
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False) # set ax1 canvas hidden so not cover up ax2

    fig.tight_layout()
    return fig

def calcSubsegmentRanking(df, dfOutcome, configs):
    capp_config = configs.get('capp_config')

    list_subseg = capp_config['SUB_SEGMENTS']
    dfSubAR = pd.DataFrame(
        columns = pd.MultiIndex.from_product([['Risk Ranking (AR)'], dfOutcome.columns]), 
        index= pd.MultiIndex(levels=([],[]), codes=([],[]), names=['Subsegment', 'SubsegName'])
    )

    dfSubARBadCnt = dfSubAR.copy()
    col_level0 = f'# of {capp_config.get("BAD_NAME")}'
    dfSubARBadCnt.rename(columns={'Risk Ranking (AR)': col_level0}, inplace=True)

    dfSubARCoutcome = dfSubAR.copy()
    dfSubARCoutcome.rename(columns={'Risk Ranking (AR)': 'Outcome'}, inplace=True)


    for subseg_type in list_subseg:
        field = subseg_type['field']
        for subseg in df[df[field].notnull()][field].unique():
            dfSubseg = df[df[field] == subseg]
    #         print(field, subseg)
            dfSubOutcome, _, _, _ = calcRiskRankingTable(dfSubseg, configs)
            dfSubAR.loc[(subseg_type['name'], subseg),'Risk Ranking (AR)'] = dfSubOutcome.loc['AR'].values
            dfSubARBadCnt.loc[(subseg_type['name'], subseg), col_level0] = dfSubOutcome.loc[col_level0].values
            dfSubARCoutcome.loc[(subseg_type['name'], subseg), 'Outcome'] = dfSubOutcome.loc['Outcome'].values
            
    dfSubsegTest = pd.concat([dfSubAR, dfSubARBadCnt, dfSubARCoutcome], axis=1)

    dfSubsegTest.index.names = (None, None)
    dfSubsegTest.columns.names = (None, None)
    outcomeHighlight = {'GREEN': 'outcome-complied', 'YELLOW': 'outcome-moderate', 'RED': 'outcome-significant'}
    dfHighlight = dfSubsegTest.applymap(lambda x: outcomeHighlight.get(x, ''))

    formatFrontend = formatMultiIndexTable(cols=dfSubsegTest.columns,
                                        firstIndexNames=['Risk Ranking (AR)','# of Default', 'Outcome'],
                                        formats = ['{:,.1%}', '{:,.0f}', None])

    sOutcome = docTable(dfSubsegTest, 
            show_index=True,
            caption='Risk Ranking Outcome',
            table_type = 'text-center',
            numbers_format=formatFrontend, 
            highlight_cells=dfHighlight
            )
    return sOutcome

def calcFactorAR(df, factorSeq, configs):
    capp_config = configs.get('capp_config')
    factor_config = configs.get('factor_config')

    bad_ind = capp_config['BAD']
    factorConfig = factor_config['factors'][factorSeq-1]
    curSeq = factorConfig['factorSeq']
    curName = factorConfig['factorName']
    curFV = factorConfig['factorValue']
    curFS = factorConfig['factorScore']
    curBuckets = factorConfig['factorBuckets']
    curBucketCut = factorConfig['factorBucketCut']

    dfFactor = df[[capp_config['KEY_FIELD']] + capp_config['COHORT_GROUP'] + [curFV, curFS, bad_ind]].copy()

    if dfFactor[curFV].dtype == 'object':
            dfFactor[curFV] = dfFactor[curFV].fillna('Missing')
    else:
            dfFactor[curFV] = dfFactor[curFV].fillna(-99999)


    if type(curBucketCut) == list: # using list to cut into bucket for numeric factor
        dfFactor[curFV + '_bin'] = pd.cut(
            dfFactor[curFV],
            bins = curBucketCut,
            labels = curBuckets,
            ordered = False
            )
    elif type(curBucketCut) == dict: # using a dict to map categorical factor
        facMap = {}
        othersKey = ''
        for k,vv in curBucketCut.items():
            if type(vv) == list:
                for v in vv:
                    facMap[v] = k
            elif ((type(vv) == str) & (vv != 'OTHERS')) or type(vv) == int :
                facMap[vv] = k
            else:
                othersKey = k

        dfFactor[curFV + '_bin'] = dfFactor[curFV].map(lambda x: facMap.get(x, othersKey))
    else: # using raw value for bucketing
        dfFactor[curFV + '_bin'] = dfFactor[curFV]


    dfCount =  dfFactor \
        .pivot_table(
            index = [curFV + '_bin',  curFS],
            columns = capp_config['COHORT_GROUP'],
            aggfunc = {bad_ind: ['count', 'sum']},
            observed=True,
        ).sort_index()
    
    dfCount.columns = dfCount.columns.droplevel(0)
    dfCount.rename(columns = {'count': 'Count', 'sum': capp_config.get('BAD_NAME')}, inplace=True)
    dfCount.index.names = ('Bucket', 'Score')
    
    bucket = dfCount.index.get_level_values(1)
    dfOutcome, styledOutcome = getRiskRankingOutcome(dfCount, bucket=bucket, isFactorLevel=True, configs=configs)
    dfOutcome.insert(0, 'Factor Name', curName)

    for col in dfCount.columns:
        if col[0] == 'Count':
            total = dfCount[col].sum()
            dfCount[('Distribution', col[1], col[2])] = dfCount[col]/total
        if col[0] == capp_config['BAD_NAME']:
            dfCount[(capp_config['BADRATE_NAME'], col[1], col[2])] = dfCount[col]/dfCount[('Count', col[1], col[2])]

    # begin styling the backend table
    styledBackend = dfCount.copy()
    styledBackend.columns.names = [None, None, None]

    styledBackend.loc[('Total', ''),:] = styledBackend.sum(axis=0)
    badrates = (styledBackend.loc[('Total', ''), capp_config['BAD_NAME']] / styledBackend.loc[('Total', ''), 'Count']).values
    styledBackend.loc[('Total', ''),capp_config['BADRATE_NAME']] = badrates

    styledBackend.reset_index(col_level=2, inplace=True)
    styledBackend = docTable(styledBackend,
#                             show_index = True,
                            table_type = 'text-center ',
                            caption = f'{curName} Factor - Distribution & Bad Rate',
                            numbers_format=formatMultiIndexTable(cols=styledBackend.columns, 
                                              firstIndexNames=['Count', capp_config['BAD_NAME'], 'Distribution', capp_config['BADRATE_NAME']],
                                              formats =['{:,.0f}', '{:,.0f}', '{:,.1%}','{:,.2%}']
                                             ))
    # end styling the backend table

    return {
        'factorSeq': curSeq,
        'factorName': curName,
        'factorAR': dfOutcome,
        'factorDist': styledBackend,
        'factorPerformRank': curSeq,
    }

def calcMultiFactorAR(df, configs):
    capp_config = configs.get('capp_config')
    factor_config = configs.get('factor_config')

    multiFactors = []
#     dfMultiFactorAR = pd.DataFrame(columns=)
    for factor in factor_config['factors']:
        multiFactors.append(calcFactorAR(df, factor['factorSeq'], configs))
        
    ARs = pd.concat([factor.get('factorAR').loc['AR':'AR'] for factor in multiFactors]).reset_index(drop=True)
    ARs.set_index('Factor Name', inplace=True)
    ARs.columns = pd.MultiIndex.from_product([['Risk Ranking (AR)'], ARs.columns])
    
    outcomes = pd.concat([factor.get('factorAR').loc['Outcome':'Outcome'] for factor in multiFactors]).reset_index(drop=True)
    outcomes.set_index('Factor Name', inplace=True)
    outcomes.columns = pd.MultiIndex.from_product([['Test Outcome'], outcomes.columns])
    
    dfARs = pd.concat([ARs, outcomes], axis=1)
    
    outcomeHighlight = {'GREEN': 'outcome-complied', 'YELLOW': 'outcome-moderate', 'RED': 'outcome-significant'}
    dfHighlight = dfARs.applymap(lambda x: outcomeHighlight.get(x, ''))
    
    dfARs.columns.name = None
    dfARs.index.names = [None]
    
    sOutcome = docTable(dfARs, 
         show_index=True,
         caption='Factors Risk Ranking Performance',
         table_type = 'text-center index-left',
         numbers_format='{:,.1%}', 
         highlight_cells=dfHighlight
        )
    
        
    return sOutcome, multiFactors
