import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def standardize_data(df):
    features = df.columns
    x = df.loc[:, features].values
    return StandardScaler().fit_transform(x)  # Standardizing the features

def perform_pca(x, n_components=3):
    pca = PCA(n_components=n_components)  # n_components is the number of principal components you want
    principalComponents = pca.fit_transform(x)
    loadings = pca.components_
    return principalComponents, loadings

def create_dataframe(principalComponents, loadings, features):
    loadings_df = pd.DataFrame(loadings, columns=features, index=['PC1', 'PC2', 'PC3'])
    principalDf = pd.DataFrame(data = principalComponents, columns = ["PC1", "PC2", "PC3"])
    return principalDf, loadings_df

def plot_data(principalDf, loadings_df):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Create 1 row, 2 columns of subplots

    # Plot heatmap on the first subplot
    sns.heatmap(loadings_df, cmap='coolwarm', ax=axs[0])

    # Create 3D scatter plot on the second subplot
    ax = fig.add_subplot(122, projection='3d')

    x = principalDf["PC1"]
    y = principalDf[ "PC2"]
    z = principalDf[ "PC3"]

    ax.scatter(x, y, z)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Scatter Plot of Principal Components')

    plt.show()



def pca_analysis(df):
    x = standardize_data(df)
    principalComponents, loadings = perform_pca(x)
    principalDf, loadings_df = create_dataframe(principalComponents, loadings, df.columns)
    plot_data(principalDf, loadings_df)



def visualize_data(data_csv):
    original_data = pd.read_csv(data_csv)

    df = original_data.drop( ["V54","V55"],axis=1 )
    #print(original_data.shape)
    #print(df.shape)

    #print(df.loc[13:14, ['V54']])


    small_data = original_data.loc[:, ["V02",
         "V03",
         "V21",
         "V22",
         "V34",
         "V35",
         "V38",
         "V39",
         "V44",
         "V47",
         "V48",
         "V57"]]

    pca_analysis(small_data)


def main():
    # TODO: Implement the main logic of the program
    visualize_data('CESR.csv')
    

if __name__ == "__main__":
    main()
