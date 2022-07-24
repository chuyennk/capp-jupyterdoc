from base64 import b64decode, b64encode
import pandas as pd
import numpy as np
from scipy.stats import binom
from typing import Dict, List, Tuple, Callable
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import seaborn as sns
from io import BytesIO
from base64 import b64encode, b64decode


from IPython.core.display import HTML, Javascript, display
from pandas.io.formats.style import Styler

# from mylibs.configs import capp_config, waterfall_config, scoreband_config, factor_config, CA_threshold
# from mylibs.configs import get_config


def readModelInfo(docname='', tab='Doc-Info'):
    df = pd.read_excel(docname, sheet_name=tab)
    df.fillna('', inplace=True)
    
    return df

# Define a utils to return a dict with columns format
def formatMultiIndexTable(cols: List, firstIndexNames: List, formats: List) -> Dict:
    strformatter = {}
    for idx, idxName in enumerate(firstIndexNames):
        for col in cols:
            if col[0] == idxName:
                strformatter[col] = formats[idx]
            
    return strformatter

def docTable(
    df,
    caption = '',
    show_index = False,
    no_header = False,
    no_border=False,
    pre_wrap_data=False,
    numbers_format='',
    table_type='',
    highlight_cells=None,
    ):
    
    styler = Styler(df, uuid_len=0, cell_ids=False)
    styler.set_caption(caption)
    
    table_classes = f'ctable {table_type} '
    if show_index == False:
        # styler.hide(axis='index')
        styler.hide_index()
        
    if no_header:
        table_classes += "no-header "
        
    if no_border:
        table_classes += "no-border"
        
    if pre_wrap_data:
        table_classes += "pre-wrap-data "

        
    if isinstance(numbers_format, dict):
        styler.format(numbers_format, na_rep='NA')
    elif isinstance(numbers_format, str) and len(numbers_format)>0:
        styler.format(lambda x: x if isinstance(x, str) else numbers_format.format(x), na_rep='NA')
    elif numbers_format != '':
        styler.format(numbers_format, na_rep='NA')
        
    styler.set_table_attributes(f'class="{table_classes}"')

    if highlight_cells is not None:
        styler = table_highlight(styler, highlight_cells)

    return styler

def table_highlight(styledTable: Styler, dfHighlight: pd.DataFrame) -> Styler:
    """
    This function is used to highlight the table cells.
    """

    return styledTable.set_td_classes(dfHighlight)


def create_and_excecute_code_cell(cellType='code', code=''):
    """
    Create a code cell and add it to the notebook.
    """
    return Javascript(f"""
        var cell = IPython.notebook.insert_cell_below('{cellType}');
        cell.set_text('{code}');
        cell.execute();
    """)
# create_and_excecute_code_cell("# Test")


def getWaterfallTable(df, configs):
    """
    This function is used to generate a waterfall table.
    It use the waterfall_config to determin the waterfall field, to gether with a list of groupby fields and a count field.
    It is also combined with the another dataframe for good/bad count.

    :Parameters:
        df: a DataFrame with these columns: waterfal_ind, sample, cohort, goodbad fields and a none null field to count the samples
    """
    capp_config = configs.get('capp_config')
    waterfall_config = configs.get('waterfall_config')

    dfWaterfall1 = df.pivot_table(
        index = waterfall_config.get('indexField'),
        columns = capp_config.get('COHORT_GROUP'),
        values = capp_config.get('KEY_FIELD'), 
        aggfunc = 'count',
        margins = True
        )

    dfWaterfall2 = df[
        df[waterfall_config.get('indexField')] == waterfall_config.get('backendValue')
                         ].pivot_table(
                            index = waterfall_config.get('BAD'),
                            columns = capp_config.get('COHORT_GROUP'),
                            values = capp_config.get('KEY_FIELD'), 
                            aggfunc='count',
                            margins = True
                        )

    dfWaterfall2 = dfWaterfall2.loc[~dfWaterfall2.index.isin(['All'])]
    
    dfWaterfall = pd.concat([dfWaterfall1, dfWaterfall2])[dfWaterfall1.columns].fillna(0)
    dfWaterfall = dfWaterfall.rename(index=waterfall_config.get('category')).sort_index(axis=0)
    dfWaterfall.columns.names = [None,'Category']
    
    return dfWaterfall

def getFrontendPSI(df, configs):
    """
    This function is to compute the frontend PSI and AHI tests.

    :Parameters:
        df: a DataFrame with frontend base info, including the Score, Cohort group (i.e fields for cohort groupping).

    :Returns: the frontend distribution dataframe and the outcome table for PSI & AHI.

    """
    capp_config = configs.get('capp_config')
    scoreband_config = configs.get('scoreband_config')

    df['scoreband'] = pd.cut(df[capp_config['SCORE']], scoreband_config['scoreband'])

    dfFrontend = df.pivot_table(
        index = 'scoreband',
        columns = capp_config['COHORT_GROUP'],
        values = capp_config['SCORE'], 
        aggfunc='count',
        observed = True
    ).fillna(0)
    
    dfFrontend.columns = pd.MultiIndex.from_tuples([('Count', *col) for col in dfFrontend.columns])
    
    dfDist = dfFrontend.copy()
    dfIV = dfFrontend.drop(dfFrontend.columns[0], axis=1).copy()
    listAHI = []
    
    for col in dfFrontend.columns:
        # Compute distribution by taking each scoreband count divide for the total, by each cohort
        colsum = dfFrontend[col].sum()        
        # check if there is account in the cohort. If not, just assign 1 account
        dfDist[col] = np.where(dfFrontend[col]>0, dfFrontend[col]/colsum, 1/colsum)
        listAHI.append((np.sum(dfDist[col] * dfDist[col]) - 1/dfDist[col].count())/(1 - 1/dfDist[col].count()))
        
        # Compute IV by take (review distr - base distr) * log(review distr / base distr)
        if col[2] == capp_config['SAMPLE_BASE_VALUE']:
            baselineDist = dfDist[col]
        else:
            diffarr = dfDist[col] - baselineDist
            logarr = np.log(dfDist[col]/baselineDist)
            dfIV[col] = diffarr*logarr
    
    dfDist = dfDist.rename(columns={'Count': 'Distribution'})
    dfIV = dfIV.rename(columns={'Count': 'IV'})

    dfFrontend = pd.concat([dfFrontend, dfDist, dfIV], axis=1)
    dfFrontend.index.name=None
    dfFrontend.loc['Total',:] = dfFrontend.sum(axis=0) #sum all counts, distr and IV to a Total row
    
    dfPSI = dfFrontend.loc['Total':,'IV'].rename(index={'Total':'PSI'})
#     dfPSI.loc['PSI Outcome'] = []
    dfAHI = pd.DataFrame([listAHI], index = ['AHI'], columns=dfFrontend['Count'].columns)
    dfOutcome = pd.concat([dfPSI, dfAHI])

    return dfFrontend, dfOutcome

def computeFactorCA(df, factorSeq, configs):
    """
    This function takes a review dataframe and a factor config to compute the Characteristic Assessment (CA)

    :Parameters:
        df: a DataFrame with those columns configured in the capp_config and factor_config
        factor: a factor config dict in factor_config

    :Returns:
        a Dict with factor name, seq, factor distribution table and its CA info.
    """
    capp_config = configs.get('capp_config')
    factor_config = configs.get('factor_config')


    factorConfig = factor_config['factors'][factorSeq-1]
    curSeq = factorConfig['factorSeq']
    curName = factorConfig['factorName']
    curFV = factorConfig['factorValue']
    curFS = factorConfig['factorScore']
    curBuckets = factorConfig['factorBuckets']
    curBucketCut = factorConfig['factorBucketCut']

    dfFactor = df[[capp_config['KEY_FIELD']] + capp_config['COHORT_GROUP'] + [curFV, curFS]].copy()

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
            elif type(vv) == dict:
                vv = list(range(vv['from'], vv['to']+1))
                for v in vv:
                    facMap[v] = k
            else:
                othersKey = k

        dfFactor[curFV + '_bin'] = dfFactor[curFV].map(lambda x: facMap.get(x, othersKey))
    else: # using raw value for bucketing
        dfFactor[curFV + '_bin'] = dfFactor[curFV]


    dfCount =  dfFactor \
        .pivot_table(
            index = [curFV + '_bin',  curFS],
            columns = capp_config['COHORT_GROUP'],
            aggfunc = {capp_config['KEY_FIELD']: 'count'},
            observed=True,
        ).sort_index()

    dfCount.columns = dfCount.columns.droplevel(1) # remove the sample level (i.e. Dev, Review.)
    dfCount.rename(columns={capp_config['KEY_FIELD']:'Count'}, inplace=True)

    dfDist = dfCount.copy()
    dfAvgScore = dfCount.drop(dfCount.columns[0],axis=1).copy()

    arrBaseScore = None
    arrAvgScore = None

    for col in dfDist.columns:
        dfDist[col] = dfDist[col]/dfDist[col].sum()
        if col[1] == capp_config['SAMPLE_BASE_VALUE']:
            arrBaseScore = dfDist[col] 
        else:
            arrAvgScore = dfDist[col]
            dfAvgScore[col] = (arrAvgScore - arrBaseScore)* dfDist.index.get_level_values(1)

    dfDist.rename(columns={'Count':'Distribution'}, inplace=True)
    dfAvgScore.rename(columns={'Count':'Score Changes'}, inplace=True)

    dfFactorTable = pd.concat([dfCount, dfDist, dfAvgScore], axis=1)
    dfFactorTable.loc[('Total',''),:] = dfFactorTable.sum(axis=0) 
    dfFactorTable.index.name = None
    dfFactorTable.columns.name = None

    dfFactorCA = dfFactorTable.loc['Total','Score Changes'].reset_index(drop=True)
    dfFactorCA = dfFactorCA[[col for col in dfFactorCA.columns if col != capp_config['SAMPLE_BASE_VALUE']]]
    dfFactorCA.columns.name=None
    dfFactorCA['Max_Abs'] = dfFactorCA.abs().max(axis=1)
    dfFactorCA.insert(0, 'Seq', curSeq)
    dfFactorCA.insert(0,'Factor Name', curName)
    
    factorCA = {
        'factorSeq': curSeq,
        'factorName': curName,
        'factorTable': dfFactorTable,
        'factorCA': dfFactorCA
        
    }
    return factorCA

def computeMultiFactorCA(df, configs):

    factor_config = configs.get('factor_config')
    CA_threshold = configs.get('CA_threshold')
    
    factorsOutcome = []
    for factor in factor_config['factors']:
        factorsOutcome.append(computeFactorCA(df, factor['factorSeq'], configs))

    dfCA = pd.concat([item['factorCA'] for item in factorsOutcome]).reset_index(drop=True)
    dfCA.set_index('Factor Name', inplace=True)
    dfCA['Rank'] = dfCA['Max_Abs'].rank(ascending=False)

    dfCA.index.name=None
    
        
    # get a color table based on CA threshold for highlight outcome
    dfColor = dfCA.copy()
    dfColor.loc[dfCA['Max_Abs'].abs() >= CA_threshold['breached']] = 'outcome-significant'
    dfColor.loc[(dfCA['Rank'] <= CA_threshold['top_factor']) & (dfCA['Max_Abs'].abs() < CA_threshold['breached'])] = 'outcome-moderate'

    syledCAtable = docTable(dfCA.iloc[:, 1:-2], 
                        caption='Score Change', 
                        highlight_cells=dfColor, 
                        show_index=True, 
                        numbers_format='{:,.1f}',
                        table_type='index-left',
                        # no_border=True
                       )
    
    def stylingFactor(factor):
        df = factor['factorTable']
        df.index.names = ['Bucket', 'Score']
        df.reset_index(col_level=1, inplace=True)
        df.columns.names = [None, None]

        numbers_format = formatMultiIndexTable(cols=df.columns, 
                                                  firstIndexNames=['Count','Distribution', 'Score Changes'],
                                                  formats =['{:,.0f}','{:,.1%}','{:,.2f}']
                                                )
        return docTable(df,
                        caption = factor['factorName'],
                        numbers_format = numbers_format,
                        table_type='text-center col0-left'
                       )
    
    for factor in factorsOutcome:
        factor['factorChangeRank'] = dfCA.loc[dfCA['Seq']==factor['factorSeq'], 'Rank'].values[0]
        factor['styledTable'] = stylingFactor(factor)
    
    return syledCAtable, factorsOutcome

def getStyledTable(factorsOutcome, seq=1, rank=None):
    """
    Get styled table for the factor with the given seq and rank

    :param 
        factorsOutcome: list of factors outcome in dict format, with keys: factorSeq & factorChangeRank
        seq: factor seq
        rank: factor rank

    :return the styled table
    """
    if rank is None:
        for factor in factorsOutcome:
            if factor['factorSeq'] == seq:
                return factor['styledTable']
    else:
        for factor in factorsOutcome:
            if factor['factorChangeRank'] == rank:
                return factor['styledTable']

def calcCurveShape(df):
    ODR_MIN = 0.0000000001
    ODR_MAX = 1

    CurveshapeThreshold = {
        'AcceptDiscrepancy': 0.1,
        'SignificantLevel': 0.1,
        'RedExcessDeviation': 2,
        'RedExcessDeviationPct': 0.2,
        'YellowExcessDeviation': 1,
        'YellowExcessDeviationPct': 0.1,
    }


    df['ODR'] = df['Def']/df['Obs']

    impliedPD = (df['PD']*df['Obs']).sum()/df['Obs'].sum()
    impliedGB = (1 - impliedPD)/impliedPD
    actualGB = (df['Obs'] - df['Def']).sum()/df['Def'].sum()
    scalar = impliedGB/actualGB

    # Compute reiteratedly the scalar to get have the scaledODR reached impliedPD
    df['ODR_adj_pct'] = df['Def']/(df['Obs']*scalar + df['Def']*(1-scalar))
    while True:
        scaledODR = (df['ODR_adj_pct']*df['Obs'])*sum()/df['Obs'].sum()
        if scaledODR/impliedPD >= 0.99999 and scaledODR/impliedPD <= 1.00001:
            break
        scalar = scalar * scaledODR/impliedPD
        df['ODR_adj_pct'] = df['Def']/(df['Obs']*scalar + df['Def']*(1-scalar))

    df['ODR_adj_pct'] = np.clip(df['ODR_adj_pct'], ODR_MIN, ODR_MAX)

    df['PD_up_pct'] = (1 + CurveshapeThreshold.get('AcceptDiscrepancy'))*df['PD']
    df['PD_down_pct'] = (1 - CurveshapeThreshold.get('AcceptDiscrepancy'))*df['PD']

    df['PD_up'] = np.log(df['PD_up_pct'])
    df['PD_down'] = np.log(df['PD_down_pct'])

    df['ODR_max'] = np.log(np.clip(binom.ppf(1 - CurveshapeThreshold.get('SignificantLevel')/2, df['Obs'], df['PD_up_pct']/df['Obs']), ODR_MIN, ODR_MAX))
    df['ODR_min'] = np.log(np.clip(binom.ppf(CurveshapeThreshold.get('SignificantLevel')/2, df['Obs'], df['PD_down_pct']/df['Obs']), ODR_MIN, ODR_MAX))

    df['Breached'] = np.where((df['ODR_adj'] > df['ODR_min']) & (df['ODR_adj'] < df['ODR_max']), 0, 1)
    df['ODR_red'] = np.where(df['Breached']==1, df['ODR_adj'], -20)

    errBucketCnt = df['PD'].count()
    errExpected = errBucketCnt * CurveshapeThreshold.get('SignificantLevel')
    errExcess = round(df[df['Breached']==1]['PD'].count() - errExpected, 2)
    errExcessPct = round(errExcess/errBucketCnt, 2)

    if (errExcess>=CurveshapeThreshold.get('RedExcessDeviation') & errExcessPct>=CurveshapeThreshold.get('RedExcessDeviationPct')):
        Outcome = 'Red'
    elif (errExcess>=CurveshapeThreshold.get('YellowExcessDeviation') & errExcessPct>=CurveshapeThreshold.get('YellowExcessDeviationPct')):
        Outcome = 'Yellow'
    else:
        Outcome = 'Green'

    return df, Outcome

def getCurveshapeChart(df, outcome):
    bucket = df['PD'].apply(lambda x: '{:,.2%}'.format(x))
    DR_adj = df['ODR_adj']
    DR_adj_red = df['ODR_adj_red']
    DR_max = df['ODR_max']
    DR_min = df['ODR_min']
    PD_up = df['PD_up']
    PD_down = df['PD_down']
    Obs = df['Obs']

    darkgrey = '#929292'
    lightgrey = '#CDCDCD'
    alpha1 = 0.8
    alpha2 = 1

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()
    ax1.set(ylim = (-8,0))

    sns.barplot(x=bucket, y=Obs, color='blue', alpha=alpha1, ax=ax2)

    sns.scatterplot(x=bucket, y=DR_adj_red, color='red', s=100, zorder=10, ax=ax1)
    sns.scatterplot(x=bucket, y=DR_adj, color='black', s=100, zorder=9, ax=ax1)

    sns.lineplot(x=bucket.index, y=DR_max, color=darkgrey, alpha=alpha1, linewidth=0.1, ax=ax1)
    sns.lineplot(x=bucket.index, y=PD_up, color=darkgrey, alpha=alpha1, linewidth=0.1, ax=ax1)
    sns.lineplot(x=bucket.index, y=PD_down, color=darkgrey, alpha=alpha1, linewidth=0.1, ax=ax1)
    sns.lineplot(x=bucket.index, y=DR_min, color=darkgrey, alpha=alpha1, linewidth=0.1, ax=ax1)
    ax1.fill_between(x=bucket.index, y1 = DR_max, y2=PD_up, where=DR_max >= PD_up, color=darkgrey, alpha=alpha1)
    ax1.fill_between(x=bucket.index, y1 = PD_down, y2=DR_min, where=PD_down >= DR_min, color=darkgrey, alpha=alpha1)
    ax1.fill_between(x=bucket.index, y1 = PD_up, y2=PD_down, where=PD_up >= PD_down, color=lightgrey, alpha=alpha1)
    ax1.set(xlabal = None, ylabel = 'Log(Default Rate)')
    ax2.set(xlabel=None, ylabel='Number of Observations')

    ax2.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) # format y-axis to thousands

    ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    ax1.patch.set_visible(False) # hide the 'canvas' so not cover up ax2

    sns.despine(left=False, bottom=False, right=False, top=False) # remove the top spines

    outcomeColor = 'lightgreen' if outcome == 'Green' else 'yellow' if outcome == 'Yellow' else 'red'
    htmlBreached = 'No bucket breached' if df['Breached'].sum() == 0 else \
                                            df[['Bucket', 'PD', 'Obs', 'Def', 'ODR', 'ODR_adj_pct']][df['Breached']==1].style.format(
                                                {'PD': '{:,.2%}',
                                                'ODR': '{:,.2%}',
                                                'ODR_adj_pct': '{:,.2%}',
                                                'Obs': '{:,.0f}',
                                                }
                                            ).render()
    figdata = BytesIO()
    fig.savefig(figdata, format='png', dpi=300)
    html = f"""
    <table>
        <tr>
            <td>Curveshape Test for CUL portfolio</td>
            <td style="background-color: {outcomeColor}">{outcome}</td>
            <td>Buckets Breached</td>
        </tr>
        <tr>
            <td colspan="2"><img width="500px" height="300px" src="data:image/png;base64,{b64encode(figdata.getvalue()).decode()}"/></td>
            <td>{htmlBreached}</td>
        </tr>
    </table>
    """

    HTML(html)
                                            

