#starting 
#project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv("TB_Burden_Country.csv

# Check basic info
print(data.info())
print(data.shape)
print(data.head())

# Check and visualize missing values
sns.heatmap(data.isnull(), cbar=False, cmap='YlGnBu')
plt.title('Missing Values Heatmap', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# Drop columns with many NaNs
data = data.drop(['Method to derive TBHIV estimates'], axis=1)

# Drop unnecessary codes
data = data.drop([
    'ISO 2-character country/territory code',
    'ISO 3-character country/territory code',
    'ISO numeric country/territory code'], axis=1)

# Country-level stats
Country_stats = data.groupby("Country or territory name")[[
    "Estimated prevalence of TB (all forms) per 100 000 population",
    "Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population"
]].mean().dropna()

top_prevalence = Country_stats.sort_values(
    by="Estimated prevalence of TB (all forms) per 100 000 population", ascending=False).head(10)

top_mortality = Country_stats.sort_values(
    by="Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population", ascending=False).head(10)

# Bar plots - top countries
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.barplot(data=top_prevalence.reset_index(),
            x="Estimated prevalence of TB (all forms) per 100 000 population",
            y="Country or territory name",
            palette="magma", ax=axes[0])
axes[0].set_title("Top 10 Countries by TB Prevalence", fontsize=12)
axes[0].set_xlabel("Prevalence", fontsize=10)
axes[0].set_ylabel("Country", fontsize=10)

sns.barplot(data=top_mortality.reset_index(),
            x="Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population",
            y="Country or territory name",
            palette="coolwarm", ax=axes[1])
axes[1].set_title("Top 10 Countries by TB Mortality", fontsize=12)
axes[1].set_xlabel("Mortality", fontsize=10)
axes[1].set_ylabel("Country", fontsize=10)

plt.tight_layout()
plt.show()

# Correlation heatmap
sns.heatmap(Country_stats.corr(), annot=True, cmap="YlOrBr", fmt=".2f")
plt.title("Correlation of TB Metrics (Country Level)", fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# Region-wise analysis
data3 = pd.read_csv("TB_Burden_Country.csv")
data3.columns = data3.columns.str.strip()

region_stats = data3.groupby("Region")[[
    "Estimated prevalence of TB (all forms) per 100 000 population",
    "Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population"
]].mean().dropna()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.barplot(data=region_stats.reset_index(),
            x="Estimated prevalence of TB (all forms) per 100 000 population",
            y="Region", palette="Greens", ax=axes[0])
axes[0].set_title("Avg TB Prevalence by Region", fontsize=12)
axes[0].set_xlabel("Prevalence", fontsize=10)
axes[0].set_ylabel("Region", fontsize=10)

sns.barplot(data=region_stats.reset_index(),
            x="Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population",
            y="Region", palette="Blues", ax=axes[1])
axes[1].set_title("Avg TB Mortality by Region", fontsize=12)
axes[1].set_xlabel("Mortality", fontsize=10)
axes[1].set_ylabel("Region", fontsize=10)

plt.tight_layout()
plt.show()

# Temporal trends
trends = data.groupby("Year")[[
    "Estimated prevalence of TB (all forms) per 100 000 population",
    "Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population"
]].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=trends, x="Year",
             y="Estimated prevalence of TB (all forms) per 100 000 population", label="Prevalence", marker="o", color='green')
sns.lineplot(data=trends, x="Year",
             y="Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population", label="Mortality", marker="o", color='red')
plt.title("Global TB Trends Over Time", fontsize=12)
plt.xlabel("Year", fontsize=10)
plt.ylabel("Rate per 100,000", fontsize=10)
plt.legend(title="Metric", fontsize=8)
plt.tight_layout()
plt.show()

# Correlation matrix and scatter plots
data.columns = data.columns.str.strip()
corr_data = data[[
    "Estimated prevalence of TB (all forms) per 100 000 population",
    "Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population",
    "Estimated incidence (all forms) per 100 000 population",
    "Case detection rate (all forms), percent"
]].dropna()

corr_matrix = corr_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Matrix of TB Metrics", fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# Scatter: Prevalence vs Mortality
plt.figure(figsize=(6, 5))
sns.scatterplot(data=corr_data,
                x="Estimated prevalence of TB (all forms) per 100 000 population",
                y="Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population",
                color="darkorange")
plt.title("Prevalence vs Mortality", fontsize=12)
plt.xlabel("Prevalence per 100k", fontsize=10)
plt.ylabel("Mortality per 100k", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter: Incidence vs Case Detection Rate
plt.figure(figsize=(6, 5))
sns.scatterplot(data=corr_data,
                x="Estimated incidence (all forms) per 100 000 population",
                y="Case detection rate (all forms), percent",
                color="teal")
plt.title("Incidence vs Case Detection Rate", fontsize=12)
plt.xlabel("Incidence per 100k", fontsize=10)
plt.ylabel("Case Detection Rate (%)", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Violin plot - Region-wise distribution
plt.figure(figsize=(12, 6))
sns.violinplot(data=data3,
               x="Region",
               y="Estimated prevalence of TB (all forms) per 100 000 population",
               palette="Pastel1")
plt.title("TB Prevalence Distribution by Region", fontsize=12)
plt.xlabel("WHO Region", fontsize=10)
plt.ylabel("TB Prevalence per 100k", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()
