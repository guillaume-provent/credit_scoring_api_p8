# API de scoring crédit

Ce repository contient l'API destinée à la prédiction du risque de défaut de remboursement d'un prêt par un client.
A partir d'une requête (/predict) avec un fichier JSON contenant les variables et leurs valeurs relatives à un client, l'API renvoie :
- la mention "DOSSIER ACCEPTE" ou "DOSSIER REFUSE"
- la probabilité de défaut de paiement du client
- le seuil de probabilité à partir duquel la dossier est refusé
- les valeurs SHAP relatives au client
- la valeur SHAP de base relative au modèle

Le ficher model.pkl contient le modèle de prédicition.
En cas de mise à jour du modèle, la constante THRESHOLD (seuil de classification) dans le fichier api.py devra être modifiée selon le paramétrage du modèle. 

Le repository est paramétré avec Github Actions pour un déploiement continu sur Heroku (fichier de procédure Procfile), incluant les tests unitaires du fichier test_api.py.
