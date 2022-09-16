import streamlit
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from PIL import Image
import os
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, preprocessing, metrics
import graphviz as graphviz
from dtreeviz.trees import dtreeviz
import math
import base64
import streamlit.components.v1 as components

streamlit.set_option('deprecation.showPyplotGlobalUse', False)

# import plotly.figure_factory as ff

load_dotenv()


################ All functionalities ####################

def getMean(dataframe):

    total = 0

    for datapoint in dataframe:
        total = total + datapoint

    return total / len(dataframe)


def getMode(dataframe):

    map = {}

    mode = 0
    max_freq = 0

    for datapoint in dataframe:
        if datapoint not in map:
            map[datapoint] = 1
        else:
            map[datapoint] = map[datapoint] + 1

        if map[datapoint] > max_freq:
            max_freq = map[datapoint]
            mode = datapoint

    return mode


def getMedian(dataframe):

    median = None
    sze = len(dataframe)
    data = []

    if sze == 1:
        return dataframe[0]

    for datapoint in dataframe:
        data.append(datapoint)

    if sze % 2 == 0:
        median = (data[sze // 2] + data[sze // 2 - 1]) / 2
    else:
        median = data[sze // 2]

    return median


def getRange(dataframe):
    return dataframe[len(dataframe) - 1] - dataframe[0]


def getMidrange(dataframe):
    return getRange(dataframe) / 2


def getVarience(dataframe):
    mean = getMean(dataframe)

    varience = 0

    for datapoint in dataframe:
        varience = varience + ((datapoint - mean) ** 2)

    varience = varience / len(dataframe)

    return varience


def getStandardDeviation(dataframe):

    return getVarience(dataframe) ** 0.5


def getQuartile(dataframe, n):

    sze = len(dataframe)
    if n == 2:
        return getMedian(dataframe)
    elif n == 1:
        return getMedian(dataframe.iloc[: sze // 2])
    elif n == 3:
        return getMedian(dataframe.iloc[sze // 2 + 1 :])


def descriptiveAnalysis(dataframe):

    streamlit.header("Central Tendency")
    options = []
    for attribute in dataframe.columns:
        if pd.to_numeric(dataframe[attribute], errors="coerce").notnull().all() == True:
            options.append(attribute)

    if len(options) == 0:
        streamlit.error("No numerical attribute to Analyse!")
    else:
        cols = streamlit.columns([1, 1, 1, 1])

        attribute = cols[3].selectbox(label="Select Attribute", options=options)
        precision = cols[3].slider("Precision", 0, 13)
        dataframe = dataframe.sort_values(by=[attribute])

        cols[0].metric(
            "Mean",
            round(getMean(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )
        cols[0].metric(
            "Median",
            round(getMedian(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )
        cols[1].metric(
            "Mode",
            round(getMode(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )
        cols[1].metric(
            "Midrange",
            round(getMidrange(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )
        cols[2].metric(
            "Variance",
            round(getVarience(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )
        cols[2].metric(
            "Standard Deviation",
            round(getStandardDeviation(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )

    streamlit.header("Dispersion of Data")

    if len(options) == 0:
        streamlit.error("No numerical attribute to Analyse!")
    else:
        cols = streamlit.columns([1, 1, 1, 1])
        dataframe = dataframe.sort_values(by=[attribute])

        cols[0].metric(
            "Range",
            round(getRange(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )

        cols[3].metric(
            "Interquartile Range",
            round(
                getQuartile(dataframe[attribute], 3)
                - getQuartile(dataframe[attribute], 1),
                precision,
            ),
            delta=None,
            delta_color="normal",
        )

        cols[2].metric(
            "Maximum",
            round(max(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )

        cols[1].metric(
            "Minimum",
            round(min(dataframe[attribute]), precision),
            delta=None,
            delta_color="normal",
        )

        cols[0].metric(
            "Quartile - 1",
            round(getQuartile(dataframe[attribute], 1), precision),
            delta=None,
            delta_color="normal",
        )

        cols[1].metric(
            "Quartile - 2",
            round(getQuartile(dataframe[attribute], 2), precision),
            delta=None,
            delta_color="normal",
        )

        cols[2].metric(
            "Quartile - 3",
            round(getQuartile(dataframe[attribute], 3), precision),
            delta=None,
            delta_color="normal",
        )


def visualAnalysis(dataframe):

    options = []
    for attribute in dataframe.columns:
        if pd.to_numeric(dataframe[attribute], errors="coerce").notnull().all() == True:
            options.append(attribute)

    streamlit.header("Histogram")

    if len(options) == 0:
        streamlit.error("No numerical attribute to Analyse!")
    else:
        cols = streamlit.columns([3, 1])

        attribute = cols[1].selectbox(label="Select Attribute", options=options)
        dataframe = dataframe.sort_values(by=[attribute])

        fig, ax = plt.subplots()
        plt.locator_params(nbins=15)
        plt.xlabel(attribute)
        plt.ylabel("count")
        ax.hist(dataframe[attribute])
        cols[0].pyplot(fig)
        plt.clf()

    streamlit.header("Scatter Plot")

    if len(options) == 0:
        streamlit.error("No numerical attribute to Analyse!")
    else:
        cols = streamlit.columns([3, 1])

        xlabel = cols[1].selectbox("X Axis Attribute", options)
        ylabel = cols[1].selectbox("Y Axis Attribute", options)
        plt.locator_params(nbins=10)
        plt.scatter(dataframe[xlabel], dataframe[ylabel], c="green", s=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cols[0].pyplot(plt)
        plt.clf()

    streamlit.header("Box Plot")

    if len(options) == 0:
        streamlit.error("No numerical attribute to Analyse!")
    else:
        cols = streamlit.columns([3, 1])
        dataframe = dataframe.sort_values(by=[attribute])

        indices = []
        options = []
        ind = 0
        for i in dataframe.columns:
            if dataframe.dtypes[i] != object:
                options.append(i)
                indices.append(ind)
            ind += 1

        data = dataframe.iloc[:, indices].values
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(data, labels=options)
        cols[0].pyplot(plt)


def correlationAnalysis(dataframe):

    options = []
    classColumn = None
    for attribute in dataframe.columns:
        classColumn = attribute
        if pd.to_numeric(dataframe[attribute], errors="coerce").notnull().all() == True:
            options.append(attribute)

    cols = streamlit.columns([1, 3])

    cols[0].header("Chi Sqaure Test")
    att1 = cols[0].selectbox("Select first attribute", options)

    contigency = pd.crosstab(dataframe[att1], dataframe[classColumn], margins=True)
    cols[1].subheader("Contingency/Observed Table")
    cols[1].text(contigency)

    row_len = len(dataframe[att1].unique())
    col_len = len(dataframe[classColumn].unique())
    row_sum = contigency.iloc[0:row_len, col_len].values
    exp = []
    for j in range(row_len):
        for val in contigency.iloc[row_len, 0:col_len].values:
            exp.append(val * row_sum[j] / contigency.loc["All", "All"])

    obs = []
    for j in range(row_len):
        for val in contigency.iloc[j, 0:col_len].values:
            obs.append(val)

    # Expected Table
    expArr = np.array(exp).reshape(row_len, col_len)
    degreeOfFreedom = (row_len - 1) * (col_len - 1)
    objmexp = []
    chiSquareValue = 0

    for i in range(len(obs)):
        chiSquareValue += (obs[i] - exp[i]) ** 2 / exp[i]
        objmexp.append((obs[i] - exp[i]) ** 2 / exp[i])
    objmexp = np.array(objmexp).reshape(row_len, col_len)

    # chi Square Value
    cols[0].metric(
        "Chi Square Value",
        chiSquareValue,
        delta=None,
        delta_color="normal",
    )

    # covariance
    cols[0].subheader("Select Attributes For Covaraince & Pearson")
    attr1 = cols[0].selectbox("Select 1st attribute", options)
    attr2 = cols[0].selectbox("Select 2nd attribute", options)
    data1 = dataframe[attr1]
    data2 = dataframe[attr2]

    xm = getMean(data1)
    ym = getMean(data2)
    n = len(data1)

    covariance = 0.0
    for i in range(n):
        covariance += (data1[i] - xm) * (data2[i] - ym) / (n - 1)

    cols[0].metric(
        "Covariance",
        covariance,
        delta=None,
        delta_color="normal",
    )

    # Pearson coefficient
    stdD1 = getStandardDeviation(data1)
    stdD2 = getStandardDeviation(data2)

    pearson = covariance / (stdD1 * stdD2)
    cols[0].metric(
        "Pearson Coefficient",
        pearson,
        delta=None,
        delta_color="normal",
    )


def normalisationAnalysis(dataframe):
    options = []
    for attribute in dataframe.columns:
        if pd.to_numeric(dataframe[attribute], errors="coerce").notnull().all() == True:
            options.append(attribute)

    att = streamlit.selectbox("Select Attribute For Normalization", options)

    # decimal Scaling
    streamlit.header("Decimal Scaling")
    data = dataframe[att].to_list()
    n = len(data)
    denom = pow(10, len(str(max(data))))

    decimal_scaling = []
    for val in data:
        decimal_scaling.append(val / denom)
    decimal_scaling.sort()

    # scatter plot of decimal scaling
    plt.locator_params(nbins=10)
    plt.scatter(decimal_scaling, decimal_scaling, c="green", s=5)
    plt.xlabel(att)
    plt.ylabel(att)
    # plt.rcParams['figure.figsize'] = [8, 4]
    streamlit.pyplot(plt)
    plt.clf()

    # Min-Max Scaling
    streamlit.header("Min-Max Normalization")
    xmin = min(data)
    xmax = max(data)
    lmin = 0  # local min
    lmax = 1  # local max
    minMax = []

    if xmin == xmax:
        streamlit.write("denominator became zero because min and max are same")
    else:
        for val in data:
            minMax.append((val - xmin) / (xmax - xmin) * (lmax - lmin) + lmin)

    # scatter plot of decimal scaling
    plt.locator_params(nbins=10)
    plt.scatter(minMax, minMax, c="green", s=5)
    plt.xlabel(att)
    plt.ylabel(att)

    # plt.rcParams['figure.figsize'] = [8, 4]
    streamlit.pyplot(plt)
    plt.clf()

def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}" style="width: 80%;"/>'

    # Write the HTML
    streamlit.write(html, unsafe_allow_html=True)

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth,rules):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            rules += ("\n" + ("{}if {} <= {}:".format(indent, name, threshold)))
            recurse(tree_.children_left[node], depth + 1)
            rules = rules + "\n" + ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            rules += ("\n" + ("{}return {}".format(indent, tree_.value[node])))

    recurse(0, 1, rules)

    return rules

def decisiontree(dataframe):
    columns = list(dataframe.columns)

    targetAttr = columns[-1]
    features = columns
    features.remove(targetAttr)
    x = dataframe[features]

    dataEncoder = preprocessing.LabelEncoder()
    encoded_x_data = x.apply(dataEncoder.fit_transform)
    classes = dataframe[targetAttr].unique().tolist()
    map = {}

    cnt = 1

    for c in classes:
        map[c] = cnt
        cnt = cnt+1 
    
    y = dataframe[targetAttr]  # Target variable
    y = y.replace(map)

    streamlit.header("Information Gain")
    # "leaves" (aka decision nodes) are where we get final output
    # root node is where the decision tree starts
    # Create Decision Tree classifer object
    decision_tree = DecisionTreeClassifier(criterion="entropy")
    # Train Decision Tree Classifer
    decision_tree = decision_tree.fit(encoded_x_data, y)

    viz = dtreeviz(decision_tree, encoded_x_data, y, target_name=targetAttr,
    feature_names=encoded_x_data.columns, class_names=classes)

    svg_write(viz.svg())

    streamlit.write(tree_to_code(decision_tree, features))

    streamlit.header("Gini Index")
    decision_tree = DecisionTreeClassifier(criterion="gini")
    # Train Decision Tree Classifer
    decision_tree = decision_tree.fit(encoded_x_data, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    tree.plot_tree(decision_tree, ax=ax, feature_names=features)
    plt.show()
    streamlit.pyplot(plt)


################# Streamlit Code Starts ####################


# Set all env variables
WCE_LOGO_PATH = os.getenv("WCE_LOGO_PATH")
WCE75 = os.getenv("WCE75")

wceLogo = Image.open(WCE_LOGO_PATH)

streamlit.set_page_config(
    page_title="Data Mining Tool",
    page_icon=WCE_LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
streamlit.markdown(hide_streamlit_style, unsafe_allow_html=True)

# padding_top = 0
# padding_side = 0
# padding_bottom = 0
# streamlit.markdown(
#     f""" <style>
#     .main .block-container{{
#         padding-top: {padding_top}rem;
#         padding-right: {padding_side}rem;
#         padding-left: {padding_side}rem;
#         padding-bottom: {padding_bottom}rem;
#     }} </style> """,
#     unsafe_allow_html=True,
# )

streamlit.markdown("<br />", unsafe_allow_html=True)

cols = streamlit.columns([2, 2, 8])

with cols[1]:
    streamlit.image(wceLogo, use_column_width="auto")

with cols[2]:
    streamlit.markdown(
        """<h2 style='text-align: center; color: red'>Walchand College of Engineering, Sangli</h2>
<h6 style='text-align: center; color: black'>(An Autonomous Institute)</h6>""",
        unsafe_allow_html=True,
    )
    streamlit.markdown(
        "<h2 style='text-align: center; color: black'>Data Mining Tool</h2>",
        unsafe_allow_html=True,
    )

# with cols[3]:
#     streamlit.image(wceLogo, use_column_width='auto')
streamlit.markdown("<hr />", unsafe_allow_html=True)
# streamlit.markdown("<h3 style='text-align: center;'>Login</h3>", unsafe_allow_html=True)

styles = {
    "container": {
        "margin": "0px !important",
        "padding": "0!important",
        "align-items": "stretch",
        "background-color": "#fafafa",
    },
    "icon": {"color": "black", "font-size": "20px"},
    "nav-link": {
        "font-size": "20px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#eee",
    },
    "nav-link-selected": {
        "background-color": "lightblue",
        "font-size": "20px",
        "font-weight": "normal",
        "color": "black",
    },
}

with streamlit.sidebar:
    streamlit.markdown(
        """<h1>Welcome,</h1>
    <h3>Suyash Sanjay Chavan<br />2019BTECS00041</h3>""",
        unsafe_allow_html=True,
    )

    streamlit.sidebar.markdown("<hr />", unsafe_allow_html=True)

    uploaded_file = streamlit.sidebar.file_uploader(
        "Upload a data file", type=[".csv"], accept_multiple_files=False
    )

    main_option = None
    dataframe = None

    if uploaded_file is not None:

        dataframe = pd.read_csv(uploaded_file)

        main_option = option_menu(
            "",
            [
                "Data Analysis",
                "Classifier",
                "About Me",
            ],
            icons=["clipboard-data", "eyeglasses", "person-badge"],
            default_index=0,
        )

if main_option == "Data Analysis":
    selected = option_menu(
        "",
        ["Descriptive", "Visual", "Correlation", "Normalization"],
        icons=["book", "eye", "search", "ui-checks"],
        orientation="horizontal",
        default_index=0,
    )

    if selected == "Descriptive":
        descriptiveAnalysis(dataframe)
    elif selected == "Visual":
        visualAnalysis(dataframe)
    elif selected == "Correlation":
        correlationAnalysis(dataframe)
    elif selected == "Normalization":
        normalisationAnalysis(dataframe)
elif main_option == "Classifier":
    selected = option_menu(
        "",
        ["Decision Tree", "Regression"],
        icons=["diagram-2", "graph-up-arrow"],
        orientation="horizontal",
        default_index=0,
    )

    if selected == "Decision Tree":
        decisiontree(dataframe)
