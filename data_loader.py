import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Load the data
df = pd.read_parquet("/Users/jovanpopovic/Desktop/honours_project/_merged_output/merged_electricity_weather.parquet")

pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)     # show all rows
np.set_printoptions(threshold=np.inf)       # print full numpy arrays

# Seperating numeric and categorical columns
# Currently, we are only using numeric columns for analysis
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# Unneeded currently, but plan to add categorical columns later
categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

#Dropping unnecessary columns
df = df.drop(columns=["ref", "row", "meter_ui", "quarter", "aggregate_date", "aggregate_year",
                      "aggregate_month", "aggregate_day", "aggregate_hour", "error_check_day", 
                      "period_over_which_rainfall_was_measured_days", "days_of_accumulation_of_maximum_temperature",
                      "days_of_accumulation_of_minimum_temperature"])
print(df.columns)

#Setting Target
target = "delivered_value"
X = df.drop(columns=[target])
y = df[target]

def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics, parse likely datetimes, make low-cardinality objects categorical,
    and coerce binary numerics to booleans. Keeps behavior generic and fast."""
    opt = df.copy()

    # Object columns: try datetime (by name hint), then numeric, else category if low cardinality
    obj_cols = opt.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        s = opt[col]
        name = col.lower()

        # Likely datetime columns (by name only, fast and safe)
        if ("date" in name) or ("timestamp" in name):
            converted = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            if converted.notna().mean() > 0.9:
                opt[col] = converted
                continue

        # Numeric-like strings
        num_try = pd.to_numeric(s, errors="coerce")
        if num_try.notna().mean() > 0.9:
            # Choose integer vs float downcast
            if (num_try.dropna() % 1 == 0).all():
                opt[col] = pd.to_numeric(s, errors="coerce", downcast="integer")
            else:
                opt[col] = pd.to_numeric(s, errors="coerce", downcast="float")
            continue

        # Low-cardinality -> category
        nunique = s.nunique(dropna=True)
        if nunique < 50 or (nunique / max(len(s), 1) < 0.5):
            opt[col] = s.astype("category")

    # Downcast existing integer and float columns
    int_cols = opt.select_dtypes(include=["int64", "int32"]).columns
    if len(int_cols) > 0:
        opt[int_cols] = opt[int_cols].apply(pd.to_numeric, downcast="integer")

    float_cols = opt.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        opt[float_cols] = opt[float_cols].apply(pd.to_numeric, downcast="float")

    # Binary numeric columns -> boolean dtype (preserve NA with pandas BooleanDtype)
    num_cols = opt.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        s = opt[col]
        vals = pd.unique(s.dropna())
        if len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0}):
            try:
                opt[col] = s.astype("boolean") if s.isna().any() else s.astype(bool)
            except Exception:
                # Fallback silently if conversion fails
                pass

    return opt

# Apply dtype optimization and refresh helpers/targets accordingly
df = _optimize_dtypes(df)
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
X = df.drop(columns=[target])
y = df[target]

#Exploratory Data Analysis (EDA) 
def eda(df):
    #Top of data (Check)
    print("Top of data")
    print(df.head())

    #Print dTypes
    print("\nData Types")
    print(df.info())

    #Summary Statistics
    print("\nSummary Statistics")
    print(df.describe().T)

    #Missing Values
    print("\nMissing Values")
    print(df.isnull().sum())

    #Percentage of missing values
    print("\nPercentage of Missing Values")
    print(df.isnull().sum() / len(df) * 100)

    #Check for duplicates
    print("\nDuplicates")
    print(f"Duplicate rows: ",df.duplicated().sum())

# Investigating the unique values in error check, see whats going on with it
def unique_values_error_check(df):
    #Unique values in each column
    print("\nUnique Values")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    # Error Check Hour Investigation
    unique_vals = df["Error Check Hour"].unique()
    print(f"Unique values in 'Error Check Hour': {unique_vals}")
    counts = df["Error Check Hour"].value_counts()
    print(f"Counts of unique values in 'Error Check Hour': {counts}")
    ech_2_row = df.loc[df["Error Check Hour"] == 13].squeeze()
    print(ech_2_row.to_frame(name='Value'))

    # Error Check Day Investigation 
    # unique_vals = df["Error Check day"].unique()
    # print(f"Unique values in 'Error Check Day': {unique_vals}")
    # counts = df["Error Check day"].value_counts()
    # print(f"Counts of unique values in 'Error Check Day': {counts}")
    # ecd_2_row = df.loc[df["Error Check day"] == 2.0].squeeze()
    # print(ecd_2_row.to_frame(name='Value'))
    # print("Missing Row of Error Check day:")
    # missing_df = df.loc[df['Error Check day'].isna()]
    # print(missing_df.T)

    # unique_vals = df["power_zero"].unique()
    # print(f"Unique values in 'Power Zero': {unique_vals}")
    # counts = df["power_zero"].value_counts()
    # print(f"Counts of unique values in 'Power Zero': {counts}")

# Investing the Received Value column
def received_value_investigation(df):
    feature = 'RECEIVED_VALUE'
    
    # Check if the feature exists in the DataFrame
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in the DataFrame.")
        return
    
    print(f"\nAnalyzing Feature: {feature}")
    
    # Summary Statistics
    print("\nSummary Statistics:")
    print(df[feature].describe())
    
    # Check for missing values
    missing_count = df[feature].isnull().sum()
    missing_percentage = missing_count / len(df) * 100
    print(f"\nMissing Values: {missing_count} ({missing_percentage:.2f}%)")
    
    unique_vals = df[feature].unique()
    # Convert numpy array to a sorted list for easier reading, ignoring nan if desired
    unique_vals_list = sorted([val for val in unique_vals if pd.notnull(val)])
    print(f"Unique values in 'Error Check Hour': {unique_vals_list}")
    counts = df[feature].value_counts()
    print(f"Counts of unique values in 'RECEIVED_VALUE': {counts}")

    rvi_1_row = df.loc[df['RECEIVED_VALUE'] == 0.001]
    if not rvi_1_row.empty:
        print(rvi_1_row)
    else:
        print("No rows where RECEIVED_VALUE equals 0.001 were found.")
    
# Investigating Delivered_Value and Daily Energy Usage
def energy_investigation(df):
    feature = 'DELIVERED_VALUE'

    # Check if the feature exists in the DataFrame
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in the DataFrame.")
        return
    
    print(f"\nAnalyzing Feature: {feature}")
    
    # Summary Statistics
    print("\nSummary Statistics:")
    print(df[feature].describe())
    
    # Check for missing values
    missing_count = df[feature].isnull().sum()
    missing_percentage = missing_count / len(df) * 100
    print(f"\nMissing Values: {missing_count} ({missing_percentage:.2f}%)")
    
    # Outlier Detection using IQR method
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print("\nOutlier Detection (IQR Method):")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    print(f"Number of Outliers Detected: {len(outliers)}")
    
    # Plotting the distribution
    # plt.figure(figsize=(12, 5))
    
    # # Histogram
    # plt.subplot(1, 2, 1)
    # plt.hist(df[feature].dropna(), bins=30, color='skyblue', edgecolor='black')
    # plt.title(f'Histogram of {feature}')
    # plt.xlabel(feature)
    # plt.ylabel('Frequency')
    
    # # Boxplot
    # plt.subplot(1, 2, 2)
    # plt.boxplot(df[feature].dropna(), vert=False)
    # plt.title(f'Boxplot of {feature}')
    # plt.xlabel(feature)
    
    # plt.tight_layout()
    # plt.show()

# Investigating Quarter Column
def quarter_investigation(df):
    feature = 'Quarter'
    
    # Check if the feature exists in the DataFrame
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in the DataFrame.")
        return
    
    print(f"\nAnalyzing Feature: {feature}")
    
    # Summary Statistics
    print("\nSummary Statistics:")
    print(df[feature].describe())
    
    # Check for missing values
    missing_count = df[feature].isnull().sum()
    missing_percentage = missing_count / len(df) * 100
    print(f"\nMissing Values: {missing_count} ({missing_percentage:.2f}%)")
    
    unique_vals = df[feature].unique()
    # Convert numpy array to a sorted list for easier reading, ignoring nan if desired
    unique_vals_list = sorted([val for val in unique_vals if pd.notnull(val)])
    print(f"Unique values in 'Error Check Hour': {unique_vals_list}")

    unique_vals = df[feature].unique()
    print(f"Unique values in 'Quarter': {unique_vals}")
    counts = df[feature].value_counts()
    print(f"Counts of unique values in 'Quarter': {counts}")

# Investigating power_zero column
def power_zero_investigation(df):
    feature = 'power_zero'
    
    # Check if the feature exists in the DataFrame
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in the DataFrame.")
        return
    
    print(f"\nAnalyzing Feature: {feature}")
    
    # Summary Statistics
    print("\nSummary Statistics:")
    print(df[feature].describe())
    
    # Check for missing values
    missing_count = df[feature].isnull().sum()
    missing_percentage = missing_count / len(df) * 100
    print(f"\nMissing Values: {missing_count} ({missing_percentage:.2f}%)")
    

    unique_vals = df[feature].unique()
    print(f"Unique values in 'power_zero': {unique_vals}")
    counts = df[feature].value_counts()
    print(f"Counts of unique values in 'power_zero': {counts}")

    # To compare with delivered value
    counts = (df["DELIVERED_VALUE"] == 0.00).sum()
    print(f"Counts of 0 in 'delivered_value': {counts}")

# Investigating daily_energy_zero column
def daily_energy_zero_investigation(df):
    feature = 'daily_energy_zero'
    
    # Check if the feature exists in the DataFrame
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in the DataFrame.")
        return
    
    print(f"\nAnalyzing Feature: {feature}")
    
    # Summary Statistics
    print("\nSummary Statistics:")
    print(df[feature].describe())
    
    # Check for missing values
    missing_count = df[feature].isnull().sum()
    missing_percentage = missing_count / len(df) * 100
    print(f"\nMissing Values: {missing_count} ({missing_percentage:.2f}%)")

    unique_vals = df[feature].unique()
    print(f"Unique values in 'daily_power_zero': {unique_vals}")
    counts = df[feature].value_counts()
    print(f"Counts of unique values in 'daily_power_zero': {counts}")

    # To compare with delivered value
    counts = (df["Daily Energy Usage"] == 0.00).sum()
    print(f"Counts of 0 in 'daily energy usage': {counts}")

def univariate_analysis(df):
    # Univariate analysis for numeric columns
    for col in numeric_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f'Univariate Analysis of {col}')
        plt.show()

def bivariate_analysis(df):
    # Bivariate analysis for numeric columns
    for col in numeric_columns:
        if col != target:  # Avoid plotting against itself
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=df[col], y=df[target])
            plt.title(f'Bivariate Analysis of {col} vs {target}')
            plt.xlabel(col)
            plt.ylabel(target)
            plt.show()

    for col in categorical_columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.show()

    # Correlation matrix (numerical only)
    corr = df[numeric_columns].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def multivariate_analysis(df):
    # Heatmap of all numeric features (already above, but larger)
    plt.figure(figsize=(12,10))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap of Numeric Features")
    plt.show()

    # Pair plot for numeric features (can be slow for many columns)
    sns.pairplot(df[numeric_columns].sample(min(500, len(df))), diag_kind='kde')
    plt.show()
    
def feature_importance(X, y):
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit baseline model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=False).head(10).plot(kind="barh", figsize=(8,5))
    plt.title("Top 10 Feature Importances")
    plt.gca().invert_yaxis()
    plt.show()

# print("Strongest positive correlations with target:")
# print(corr[target].sort_values(ascending=False).head(10))

# print("\nStrongest negative correlations with target:")
# print(corr[target].sort_values().head(10))

# univariate_analysis(df)

eda(df)
