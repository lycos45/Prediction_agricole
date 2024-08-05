git import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
import statsmodels.formula.api as smf
import statsmodels.api as sm
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold

# Fonction de prétraitement des données
def wrangle(filepath):
    df = pd.read_csv(filepath, sep=';', na_values=["", " ", "  ", "   ", "    ", np.nan], low_memory=False)

    # Remplacer le d'un variable
    df['pratique_cultures_fourrageres'] = df['1. pratique_cultures_fourrageres']
    df.drop(columns="1. pratique_cultures_fourrageres", inplace=True)

    # Création du variable rendement et suppression des autres rendements
    def merge_columns(row):
        cols = [row['rend_moyen_arachide'], row['rend_moyen_mil'], row['rend_moyen_niebe'], row['rend_moyen_mais'],
                row['rend_moyen_sorgho'], row['rend_moyen_fonio'], row['rend_moyen_riz_irrigue'],
                row['rend_moyen_riz_pluvial']]
        non_nan_values = [v for v in cols if pd.notna(v)]

        if len(non_nan_values) == 0:
            return np.nan
        elif len(non_nan_values) >= 2:
            return 0
        else:
            return non_nan_values[0]

    df['rendement'] = df.apply(merge_columns, axis=1)
    df.drop(
        columns=['rend_moyen_arachide', 'rend_moyen_mil', 'rend_moyen_niebe', 'rend_moyen_mais', 'rend_moyen_sorgho',
                 'rend_moyen_fonio', 'rend_moyen_riz_irrigue', 'rend_moyen_riz_pluvial'], inplace=True)

    # Définir les colonnes à convertir
    numcol_convert = ['rendement', 'superficie_parcelle', 'pourcentage_superficie_pastèque', 'pourcentage_CP',
                      'quantite_semence_CP', 'quantite_semence_sub_CP', 'quantite_semence_marche_specialisee_CP ',
                      'quantite_semence_reserve_personnelle_CP', 'quantite_semence_don_CP',
                      'quantite_CP', 'pourcentage_CS', 'quantite_semence_CS', 'quantite_semence_sub_CS',
                      'quantite_semence_marche_specialisee_CS', 'quantite_semence_reserve_personnelle_CS',
                      'quantite_semence_don_CS',
                      'quantite_CS', 'quantite_NPK_epandage_avant_recolte', 'quantite_urée_epandage_avant_recolte',
                      'nombre_pieds_arachide_compte', 'nombre_pieds_arachide_recoltes',
                      'poids_total_gousses_arachide_recoltés', 'poids_moyen_arachide_en_gramme',
                      'poids_recolte_arachide', 'nombre_pieds_mil_compte', 'nombre_epis_potentiel_maturite_mil',
                      'nombre_epis_prélevé_mil', 'poids_total_graines_battage_sechage_mil', 'poids_moyen_mil_en_gramme',
                      'poids_recolte_mil',
                      'nombre_pieds_niebe_compte', 'nombre_gousses_niebe_3_pieds', 'nombre_gousses_niebe_par_pieds',
                      'nombre_total_gousses_niebe', 'nombre_gousses_niebe_preleve',
                      'poids_total_gousses_niebe_apres_egrainage', 'poids_moyen_gousses_niebe',
                      'poids_total_niebe_de_la_recolte', 'nombre_pieds_mais_compte',
                      'nombre_epis_potentiel_maturite_mais', 'nombre_epis_mais_preleve',
                      'poids_total_graines_battage_sechage_mais',
                      'poids_moyen_mais_en_gramme', 'poids_recolte_mais', 'nombre_sorgho_compte',
                      'nombre_epis_potentiel_maturite_sorgho', 'nombre_epis_sorgho_preleve',
                      'poids_total_graines_battage_sechage_sorgho', 'poids_moyen_sorgho_en_gramme',
                      'poids_recolte_sorgho', 'poids_recolte_fonio', 'nombre_pieds_riz_irrigue_compte',
                      'nombre_epis_potentiel_maturite_riz_irrigue',
                      'nombre_epis_riz_irrigue_preleve', 'poids_total_graines_battage_sechage_riz_irrigue',
                      'poids_moyen_riz_irrigue_en_gramme', 'poids_recolte_riz_irrigue',
                      'nombre_pieds_riz_pluvial_compte', 'nombre_epis_potentiel_maturite_riz_pluvial',
                      'nombre_epis_riz_pluvial_preleve', 'poids_total_graines_battage_sechage_riz_pluvial',
                      'poids_moyen_riz_pluvial_en_gramme', 'poids_recolte_riz_pluvial',
                      'superficie(ha)_cultures_fourrageres']

    catcol_convert = ['id_reg', 'culture_principale', 'irrigation', 'pastèque_fin_saison', 'rotation_cultures',
                      'culture_principale_précédente_2020_2021', 'méthode_de_culture', 'culture_secondaire',
                      'varietes_arachides_CP', 'origine_semence_CP',
                      'variete_riz_CP', 'type_semence_CP', 'mois_semis_CP', 'semaine_semis_CP', 'etat_produit_CP',
                      'mois_previsionnel_recolte_CP', 'varietes_arachides_CS', 'origine_semence_CS', 'variete_riz_CS',
                      'type_semence_CS', 'mois_semis_CS',
                      'semaine_semis_CS', 'etat_produit_CS', 'mois_previsionnel_recolte_CS',
                      'utilisation_matiere_org_avant_semis', 'matieres_organiques',
                      'utilisation_matiere_miner_avant_semis', 'matieres_minerales',
                      'prevision_epandage_engrais_min_avant_recolte',
                      'type_engrais_mineral_epandage_avant_recolte', 'utilisation_produits_phytosanitaires',
                      'utilisation_residus_alimentation_animaux', 'type_residus_alimentation_animaux', 'type_labour',
                      'type_couverture_sol_interculture',
                      'type_installation', 'pratiques_conservation_sol', 'contraintes_production',
                      'type_materiel_preparation_sol', 'type_materiel_semis', 'type_materiel_entretien_sol',
                      'type_materiel_récolte',
                      'type_culture', 'rendement_arachide', 'rendement_mil', 'rendement_niebe', 'rendement_mais',
                      'rendement_riz_irrigue', 'rendement_sorgho', 'rendement_fonio', 'rendement_riz_pluvial',
                      'pratique_cultures_fourrageres',
                      'cultures_fourrageres_exploitees', 'production_forestiere_exploitation']

    # Conversion des types
    for col in numcol_convert:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].astype('float64')

    for col in catcol_convert:
        df[col] = df[col].astype('category')

    # Supprimer les variables avec plus de 50% des données manquantes
    missing_percent = df.drop(columns="rendement").isnull().mean() * 100
    columns_to_drop = missing_percent[missing_percent > 50].index.tolist()
    df.drop(columns=columns_to_drop, inplace=True)

    # Imputation des valeurs manquantes avec HistGradientBoostingRegressor
    def impute_missing_values(data, column, model):
        data_missing = df[df[column].isnull()]
        data_not_missing = df[df[column].notnull()]

        X_missing = data_missing.drop(columns=[column])
        X_not_missing = data_not_missing.drop(columns=[column])
        y_not_missing = data_not_missing[column]

        model.fit(X_not_missing, y_not_missing)
        y_missing_pred = model.predict(X_missing)
        df.loc[df[column].isnull(), column] = y_missing_pred
        return df

    model = HistGradientBoostingRegressor()
    numeric_columns = df.select_dtypes(include=['float64']).columns.tolist()

    for column in numeric_columns:
        df = impute_missing_values(df, column, model)

    # Imputation des valeurs catégorielles avec RandomForestClassifier
    def impute_categorical_values(data, column):
        data_missing = df[df[column].isnull()]
        data_not_missing = df[df[column].notnull()]

        X_missing = data_missing.drop(columns=[column])
        X_not_missing = data_not_missing.drop(columns=[column])
        y_not_missing = data_not_missing[column]

        imputer = SimpleImputer(strategy='most_frequent')
        X_missing_imputed = imputer.fit_transform(X_missing)
        X_not_missing_imputed = imputer.transform(X_not_missing)

        model = RandomForestClassifier()
        model.fit(X_not_missing_imputed, y_not_missing)
        y_missing_pred = model.predict(X_missing_imputed)
        data.loc[data[column].isnull(), column] = y_missing_pred

        return data

    for column in df.select_dtypes(include=['category']).columns.tolist():
        df = impute_categorical_values(df, column)

    # Suppression des variables non influentes
    df.drop(columns=['pastèque_fin_saison', 'contraintes_production', 'type_materiel_entretien_sol'], inplace=True)

    # Gestion des outliers
    def imputOutlier(df, var):
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        min_val = Q1 - 1.5 * IQR
        max_val = Q3 + 1.5 * IQR

        df.loc[df[var] < min_val, var] = min_val
        df.loc[df[var] > max_val, var] = max_val
        return df

    for var in df.select_dtypes("number"):
        df = imputOutlier(df, var)

    return df

# Fonction de modélisation
def modelisation(models, x_train, y_train, x_test, y_test):
    scores = []
    for model in models:
        try:
            mod = model()
        except TypeError:
            mod = model()
        debut = time()
        mod.fit(x_train, y_train)
        fin = time()
        scores.append({"modele": type(mod).__name__,
                       "temps": fin - debut,
                       "score_train": mod.score(x_train, y_train),
                       "score_test": mod.score(x_test, y_test)}
                      )
    return pd.DataFrame(scores)

def modelisation(models, x_train, y_train, x_test, y_test, k=5):
    scores = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for model in models:
        # Instanciation
        try:
            mod = model()
        except TypeError:
            mod = model()

        # Mesurer le temps d'entraînement
        debut = time()
        mod.fit(x_train, y_train)
        fin = time()

        # Scores de validation croisée
        cv_scores = cross_val_score(mod, x_train, y_train, cv=kf, scoring='r2')

        # Ajouter les scores au tableau
        scores.append({
            "modele": type(mod).__name__,
            "temps": fin - debut,
            "score_train": mod.score(x_train, y_train),
            "score_test": mod.score(x_test, y_test),
            "cv_score_mean": cv_scores.mean(),
            "cv_score_std": cv_scores.std()
        })

    return pd.DataFrame(scores)
# Interface utilisateur Streamlit
def main():
    st.title('Application de Prédiction des Rendements Agricoles')

    # Charger les données depuis le fichier local
    filepath = "base.csv"  # Change ce chemin selon la localisation de ton fichier
    df = wrangle(filepath)

    st.write("Aperçu des données prétraitées :", df.head())
    st.write("Informations sur le DataFrame :")
    st.write(df.info())

    # Afficher les statistiques descriptives
    st.subheader('Statistiques Descriptives Univariées')
    st.write(df.select_dtypes("number").describe().T)

    st.subheader('Statistiques Descriptives des Variables Catégorielles')
    st.write(df.select_dtypes(["category", "object"]).describe().T)

    # Calcul de la matrice de corrélation et heatmap
    numeric_corr = df.select_dtypes("number").corr()
    st.subheader('Matrice de Corrélation')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(numeric_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Matrice de Corrélation - Variables Numériques')
    st.pyplot(fig)

    # Boxplots des variables numériques
    st.subheader('Boxplots des Variables Numériques')
    for var in df.select_dtypes("number"):
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=var, ax=ax)
        st.pyplot(fig)

    # ANOVA pour les variables catégorielles
    st.subheader('ANOVA pour Variables Catégorielles')
    categorical_columns = df.select_dtypes(include=['category']).columns.tolist()
    significant_variables = []
    non_significatif = []
    for categorical_var in categorical_columns:
        unique_values = df[categorical_var].nunique()
        if unique_values > 1:
            formula = f'rendement ~ C({categorical_var})'
            model = smf.ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_value = anova_table["PR(>F)"][0]

            if p_value < 0.05:
                significant_variables.append(categorical_var)
            else:
                non_significatif.append(categorical_var)

    st.write("Variables significatives :", significant_variables)
    st.write("Variables non significatives :", non_significatif)

    # Préparation des données pour la modélisation
    target = 'rendement'
    x = df.drop(columns=target)
    y = df[target]
    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=3)

    # Initialiser le scaler
    scaler = StandardScaler()
    x_train[x.select_dtypes(include="number").columns] = scaler.fit_transform(x_train[x.select_dtypes(include="number").columns])
    x_test[x.select_dtypes(include="number").columns] = scaler.transform(x_test[x.select_dtypes(include="number").columns])

    # Liste des modèles à tester
    modelList = [RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, KNeighborsRegressor, LinearRegression, SVR]
    score = modelisation(modelList, x_train, y_train, x_test, y_test)

    # Afficher les résultats de la modélisation
    st.subheader('Performances des Modèles')
    st.write(score)
    # Création de la liste de modèles
    modelList = [RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, KNeighborsRegressor,
                 LinearRegression, SVR]

    # Appel de la fonction de modélisation
    score = modelisation(modelList, x_train, y_train, x_test, y_test)

    # Aperçu des résultats
    print(score)

    # Vous pouvez également afficher les résultats avec Streamlit
    st.subheader("Score apres cross-validation")
    st.write(score)


if __name__ == "__main__":
    main()

