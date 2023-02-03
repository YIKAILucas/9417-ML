import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
totalData = pd.read_csv('featured_data.csv')
trainFeature = ['ncodpers', 'ind_empleado', 'sexo', 'age', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes',
                'tiprel_1mes', 'indresi', 'indext', 'indfall', 'ind_actividad_cliente', 'renta', 'segmento']
totalFeature = totalData[trainFeature]
corrMatt = totalFeature.corr(method='pearson')
sns.heatmap(corrMatt, annot=True)
plt.show()
