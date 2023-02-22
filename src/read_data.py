import pandas as pd


class data:

    def __init__(self, file_location="amp-parkinsons-disease-progression-prediction"):

        self.train_clinical = None
        self.train_peptides = None
        self.train_proteins = None

        self.train_clinical = pd.read_csv(file_location+'/train_clinical_data.csv')
        self.train_peptides = pd.read_csv(file_location+'/train_peptides.csv')
        self.train_proteins = pd.read_csv(file_location+'/train_proteins.csv')


    def get_updrs(self, type, meds='Both'):
        # Returns severity of one of 4 tests over some time frame
        # med determines if results are under the influence of 'On', 'Off', 'Both' medication influence

        assert type in [1,2,3,4], str(type)+' is not a valid updrs type'
        assert meds in ['On', 'Off', 'Both'], meds +' is not a valid med type'

        if meds != 'Both':
            return self.train_clinical.pivot(index='visit_month',columns='patient_id', values='updrs_'+str(type))
        else:
            return self.train_clinical[self.train_clinical['upd23b_clinical_state_on_medication'] == meds].pivot(
                index='visit_month', columns='patient_id', values='updrs_' + str(type))


data()
