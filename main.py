import numpy as np
import pandas as pd



# read data into data frame

df = pd.read_csv("./ElectionsData.csv")

#df.info()

colToInt = ["Num_of_kids_born_last_10_years","Number_of_valued_Kneset_members",\
            "Number_of_differnt_parties_voted_for","Last_school_grades","Yearly_ExpensesK",\
            "Occupation_Satisfaction"]

# print(df.head(3)[colToInt])

df.fillna('None')

df.astype('int32')[colToInt]

