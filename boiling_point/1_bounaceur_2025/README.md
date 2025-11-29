# AI-ThermPpt

AI-powered prediction of critical properties and boiling points: A  hybrid ensemble learning and QSPR approach 

## Installation

To install the necessary packages run the following command:

```bash
pip install -r requirements.txt
```

# Example

A complete example file is provided in the "Example" folder.
First, you need to load the Metamodel class to take into account the developed AI models.
Next, load the developed AI models in the form of a joblib file.
After loading the list of selected descriptors, run the Mordred code to define the complete list of descriptors for each molecule.
After a quick clean-up of the values obtained, simply run the application to obtain the thermodynamic parameters.

# Folders

Dans le dossier "data", on trouve l'ensemble des valeurs ayant servi à entrainer les modèles IA, ainsi, que les valaurs des paramètres thermodynamques obtenus avec la méthode ICAS.
Dans le dossier "train models", on y trouve les modèles IA finaux entrainées sur l'ensmeble des données, ainsi que du modèle de classe nécessaire pour interpréter les modèles IA.
Dans le dossier "yaml_file", on y trouve le fichier environnement necéssaire pour reproduire les calculs. C'est un fichier environnement sous Anaconda. Il est directement interprétable.



