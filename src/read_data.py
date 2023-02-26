import pandas as pd
import numpy as np


class data:

    def __init__(self, file_location="../amp-parkinsons-disease-progression-prediction"):

        self.train_clinical = pd.read_csv(file_location+'/train_clinical_data.csv')
        self.train_peptides = pd.read_csv(file_location+'/train_peptides.csv')
        self.train_proteins = pd.read_csv(file_location+'/train_proteins.csv')

        self.protein_names = pd.unique(self.train_proteins['UniProt'])
        self.peptide_names = pd.unique(self.train_peptides['Peptide'])

    def get_x_y_data(self):
        x = self.x_data()
        y = self.y_data()
        z = pd.merge(x, y, on='visit_id', how='inner').reset_index()
        return z[[_ for _ in x.columns if _ not in ['patient_id', 'visit_month']]], z[[_ for _ in y.columns if _ not in ['patient_id', 'visit_month']]]

    def x_data(self):

        df_all = self.train_proteins.merge(self.train_peptides[['visit_id', 'UniProt', 'Peptide', 'PeptideAbundance']],
                                     on=['visit_id', 'UniProt'], how='left')

        df_by_uniprot = df_all.groupby(['visit_id', 'UniProt'])['NPX'].mean().reset_index()
        df_by_peptide = df_all.groupby(['visit_id', 'Peptide'])['PeptideAbundance'].mean().reset_index()

        df_uniprot = df_by_uniprot.pivot(index='visit_id', columns='UniProt', values='NPX').rename_axis(
            columns=None).reset_index()
        df_peptide = df_by_peptide.pivot(index='visit_id', columns='Peptide', values='PeptideAbundance').rename_axis(
            columns=None).reset_index()

        res = df_uniprot.merge(df_peptide, on=['visit_id'])
        res[['patient_id', 'visit_month']] = res.visit_id.str.split("_", expand=True)

        return res

    def x_data_3d(self):

        df_all = self.train_proteins.merge(self.train_peptides[['visit_id', 'UniProt', 'Peptide', 'PeptideAbundance']],
                                           on=['visit_id', 'UniProt'], how='left')

        df_by_uniprot = df_all.groupby(['visit_id', 'UniProt'])['NPX'].mean().reset_index()
        df_by_peptide = df_all.groupby(['visit_id', 'Peptide'])['PeptideAbundance'].mean().reset_index()

        df_uniprot = df_by_uniprot.pivot(index='visit_id', columns='UniProt', values='NPX').rename_axis(
            columns=None).reset_index()
        df_peptide = df_by_peptide.pivot(index='visit_id', columns='Peptide', values='PeptideAbundance').rename_axis(
            columns=None).reset_index()

        df_uniprot[['patient_id', 'visit_month']] = df_uniprot.visit_id.str.split("_", expand=True)
        df_peptide[['patient_id', 'visit_month']] = df_peptide.visit_id.str.split("_", expand=True)

        df_uniprot['patient_id'] = pd.to_numeric(df_uniprot['patient_id'])
        df_uniprot['visit_month'] = pd.to_numeric(df_uniprot['visit_month'])
        df_peptide['patient_id'] = pd.to_numeric(df_peptide['patient_id'])
        df_peptide['visit_month'] = pd.to_numeric(df_peptide['visit_month'])

        df_uniprot = pd.pivot_table(df_uniprot, index='patient_id', columns=['visit_month'])
        df_uniprot = df_uniprot.reindex(sorted(df_uniprot.columns, ), axis=1)
        df_peptide = pd.pivot_table(df_peptide, index='patient_id', columns=['visit_month'])
        df_peptide = df_peptide.reindex(sorted(df_peptide.columns, ), axis=1)

        return df_uniprot, df_peptide

    def y_data(self, upd23b_clinical_state_on_medication=('On', 'Off', np.nan)):
        return self.train_clinical[self.train_clinical['upd23b_clinical_state_on_medication'].isin(upd23b_clinical_state_on_medication)]

    def y_data_3d(self, upd23b_clinical_state_on_medication=('On', 'Off', np.nan)):
        return pd.pivot_table(self.train_clinical[self.train_clinical['upd23b_clinical_state_on_medication'].isin(upd23b_clinical_state_on_medication)]
                              ,index = 'patient_id',columns =['visit_month'])
