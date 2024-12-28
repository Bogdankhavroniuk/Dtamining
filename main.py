from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN , AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from IPython.display import display




class Clasterise():


    def load_data(self, file_path):

        data = pd.read_csv(file_path)

        return data

    def analyze_dataframe(self , data):

        analysis = []

        for column in data.columns:
            col_type = data[column].dtype
            missing_count = data[column].isna().sum()

            if np.issubdtype(col_type, np.number):
                # Числові змінні
                mean_val = data[column].mean()
                median_val = data[column].median()
                std_dev = data[column].std()
                quartiles = data[column].quantile([0.25, 0.5, 0.75]).values
                q1_count = ((data[column] >= data[column].min()) & (data[column] < quartiles[0])).sum()
                q2_count = ((data[column] >= quartiles[0]) & (data[column] < quartiles[1])).sum()
                q3_count = ((data[column] >= quartiles[1]) & (data[column] < quartiles[2])).sum()
                q4_count = ((data[column] >= quartiles[2]) & (data[column] <= data[column].max())).sum()

                analysis.append({
                    'Variable Name': column,
                    'Type': 'Numeric',
                    'Missing Values': missing_count,
                    'Mean': mean_val,
                    'Median': median_val,
                    'Q1 Count': q1_count,
                    'Q2 Count': q2_count,
                    'Q3 Count': q3_count,
                    'Q4 Count': q4_count,
                    'Std Dev': std_dev
                })

            elif np.issubdtype(col_type, np.object) or np.issubdtype(col_type, pd.CategoricalDtype):
                # Факторні або текстові змінні
                unique_count = data[column].nunique()
                analysis.append({
                    'Variable Name': column,
                    'Type': 'Categorical',
                    'Missing Values': missing_count,
                    'Unique Values': unique_count,
                    'Mean': None,
                    'Median': None,
                    'Q1 Count': None,
                    'Q2 Count': None,
                    'Q3 Count': None,
                    'Q4 Count': None,
                    'Std Dev': None
                })

        # Створюємо датафрейм зі статистикою
        analysis_df = pd.DataFrame(analysis)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 2)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(analysis_df)

        sns.set_style("whitegrid")

        # Створення фігури з підграфіками
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))  # Налаштовуємо розмір фігури

        # Візуалізації
        sns.histplot(data=data, x='N', ax=axes[0, 0])
        axes[0, 0].set_title('Розподіл Азоту (N)')

        sns.histplot(data=data, x='P', ax=axes[0, 1])
        axes[0, 1].set_title('Розподіл Фосфору (P)')

        sns.histplot(data=data, x='K', ax=axes[1, 0])
        axes[1, 0].set_title('Розподіл Калію (K)')

        sns.histplot(data=data, x='temperature', ax=axes[1, 1])
        axes[1, 1].set_title('Розподіл Температури')

        sns.histplot(data=data, x='ph', ax=axes[2, 0])
        axes[2, 0].set_title('Розподіл рН')

        sns.histplot(data=data, x='rainfall', ax=axes[2, 1])
        axes[2, 1].set_title('Розподіл Опадів')

        sns.histplot(data=data, x='humidity', ax=axes[3, 0])
        axes[3, 0].set_title('Розподіл Вологісті')

        sns.countplot(data=data, x='label', ax=axes[3, 1])
        axes[3, 1].set_title('Кількість Класів (Label)')

        # Налаштування
        plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Відстань по вертикалі та горизонталі

        # Показуємо графіки
        plt.show()
        plt.show()

    def plot_clusters(self, data, labels, title, cluster_names=None , folder = None):

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=labels, cmap='viridis')


        unique_labels = sorted(set(labels))

        if cluster_names:
            legend_labels = {}
            for label in unique_labels:
                if label == -1:
                    legend_labels[label] = "Шум"
                else:
                    legend_labels[label] = cluster_names[label] if label < len(cluster_names) else f"Кластер {label}"
        else:
            legend_labels = {label: "Шум" if label == -1 else f"Кластер {label}" for label in unique_labels}

        handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[label], markersize=10,
                              markerfacecolor=scatter.cmap(scatter.norm(label))) for label in unique_labels]
        ax.legend(handles=handles, title="Кластери")

        ax.set_title(title)
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        ax.set_zlabel(data.columns[2])

        plt.savefig(f"Results/{folder}/{title}.png")

    def find_most_corelated_values(self ,data , target ):

        new_df = data.copy()

        features = new_df.drop(target, axis=1)
        target_col = 'label'
        correlations = features.corrwith(data[target_col].astype('category').cat.codes)
        top_features = correlations.abs().sort_values(ascending=False).head(3).index
        selected_features = features[top_features]
        self.selected_features = selected_features


    def visualize_data(self, data):


        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.selected_features.iloc[:, 0], self.selected_features.iloc[:, 1], self.selected_features.iloc[:, 2],
                   cmap='viridis')

        ax.set_title("Попередня візуалізація даних (топ-3 ознаки)")
        ax.set_xlabel(self.selected_features.columns[0])
        ax.set_ylabel(self.selected_features.columns[1])
        ax.set_zlabel(self.selected_features.columns[2])
        plt.show()

    def perform_clustering(self , data  , min_clusters , max_clusters):


        def dbscan_fixed_clusters(data, n_clusters):

            points = data.values
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.nan)
            eps = np.nanmax(distances) # Максимальна відстань між точками
            min_samples = 2  # Початкове значення min_samples

            while True:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

                if n_clusters_ == n_clusters:
                    break
                else:
                    eps *= 0.9  # Зменшуємо eps
                    min_samples += 1

            return labels


        new_df = data.copy()


        features = new_df.drop("label", axis=1)

        selected_features = self.selected_features

        for num_of_clusters in range (min_clusters , max_clusters+1):

          results = {}
          cluster_names = ['Кластер ' + str(n) for n in range(1 , num_of_clusters+1) ]
          # Distance-based: K-Means
          kmeans = KMeans(n_clusters= num_of_clusters)
          kmeans_labels = kmeans.fit_predict(features)
          self.plot_clusters(selected_features, kmeans_labels, f"Кластеризація K-Means {num_of_clusters} кластерів",
                           cluster_names=cluster_names , folder = str(num_of_clusters)+"_кластерів")
          results['K-Means'] = kmeans_labels

          # Density-based: DBSCAN


          dbscan_labels = dbscan_fixed_clusters(features, num_of_clusters)
          self.plot_clusters(selected_features, dbscan_labels, f"Кластеризація DBSCAN {num_of_clusters} кластерів",
                           cluster_names= cluster_names ,folder = str(num_of_clusters)+"_кластерів")
          results['DBSCAN'] = dbscan_labels


          # Model-based: Gaussian Mixture
          gmm = GaussianMixture(n_components=num_of_clusters)
          gmm_labels = gmm.fit_predict(features)
          self.plot_clusters(selected_features, gmm_labels, f"Кластеризація Gaussian Mixture {num_of_clusters} кластерів",
                           cluster_names=cluster_names , folder = str(num_of_clusters)+"_кластерів")
          results['Gaussian Mixture'] = gmm_labels

          # Grid-based: Simulated Example

          grid_clustering = AgglomerativeClustering(n_clusters=num_of_clusters)
          grid_labels = grid_clustering.fit_predict(features)
          self.plot_clusters(selected_features, grid_labels, f"Кластеризація Grid-Based (Agglomerative) {num_of_clusters} кластерів",
                           cluster_names=cluster_names ,  folder = str(num_of_clusters)+"_кластерів")
          results['Grid-Based'] = grid_labels

          # Порівняння з наданими мітками
          true_labels = data['label']
          print(f"Порівняльний аналіз результатів кластеризації для {num_of_clusters} кластерів")
          for method, labels in results.items():
             print(f"Метод: {method}, Унікальні мітки: {set(labels)}")



    def unique_values_in_column(self, data, column_name):

        unique_values = data[column_name].unique()
        print(f"Унікальні значення в колонці '{column_name}': {list(unique_values)}")
        return list(unique_values)





def test(file_path = "Sourcess/Crop_recommendation.csv"):

    clasteriser = Clasterise()
    data = clasteriser.load_data(file_path)
    clasteriser.analyze_dataframe(data)
    clasteriser.find_most_corelated_values(data, "label")
    clasteriser.visualize_data(data)
    clasteriser.perform_clustering(data , 10  ,12 )



test()

