# Design Pattern Detection using BigBird

The BigBIRD project aims to classify design patterns in software projects using state-of-the-art transformer models, specifically Google's Big BIRD model. This repository contains two notebooks:

* `BigBIRD_Classification_Head_Design_Pattern_Detection.ipynb:` This notebook uses the features extracted by BigBIRD and applies classification using classical models like Logistic Regression, Random Forest, Catboost, XGBoost, and LightGBM to detect design patterns in software projects.

* `Big_Bird_Tuning.ipynb:` This notebook uses unsupervised tuning of BigBIRD model over 1400 Android repos extracted from GitHub.

## Installation
To run these notebooks, you will need the following dependencies:

* Python 3.8+
* Jupyter Notebook
* PyTorch
* Transformers
* Scikit-Learn
* Pandas
* Numpy

* You can install these dependencies by running the following command:
```
pip install torch transformers scikit-learn pandas numpy jupyter
```


## Usage
To run either of the notebooks, simply open them in Jupyter Notebook and follow the instructions.

## License
This project is licensed under the MIT License.

## Contributing
We welcome contributions from the community. To contribute, please follow these steps:

1) Fork this repository
2) Clone the forked repository to your local machine
3) Create a new branch for your changes
4) Make your changes on this branch
5) Push your changes to your forked repository
6) Open a pull request to this repository

## Credits
This project is maintained by Usman Ahmad and Ahmad Taha.

## Contact
If you have any questions or concerns, feel free to contact us at usman.ahmad@innopolis.university.
