# CON-FOLD and FOLD-RM
The implementation details of CON-FOLD and FOLD-RM algorithm and how to use it are described here. The target of these algorithms is to learn an answer set program for a classification task. Answer set programs are logic programs that permit negation of predicates and follow the stable model semantics for interpretation. 

## Installation

Only function library for FOLD-RM:
<code>
	
	python3 -m pip install foldrm
	
</code>

With the dataset examples:
<code>
	
	git clone https://github.com/hwd404/FOLD-RM.git
	
</code>

### Prerequisites
The CON-FOLD and FOLD-RM algorithm is developed with python3.10.

## Instruction
### Data preparation

The CON-FOLD and FOLD-RM take tabular data as input, the first line for the tabular data should be the feature names of each column.
The algorithms do not have to encode the data for training. It can deal with numerical, categorical, and even mixed type features (one column contains both categorical and numerical values) directly.
However, the numerical features should be identified before loading the data, otherwise they would be dealt like categorical features (only literals with = and != would be generated).

There are many UCI example datasets that have been used to pre-populate the **data** directory. Code for preparing these datasets has already been added to datasets.py.

For example, the UCI wine dataset can be loaded with the following code:

<code>
	
    attrs = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','tot_phenols','flavanoids',
    'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','OD_of_diluted','proline']
    model = Classifier(attrs=attrs, numeric=attrs, label='label')
    data = model.load_data('data/wine/wine.csv')
    print('\n% wine dataset', np.shape(data))
    return model, data

</code>

**attrs** lists all the features needed, **nums** lists all the numerical features, **label** is the name of the output classification label, **model** is an initialized classifier object with the configuration of wine dataset. 

### Training
The CON-FOLD and FOLD-RM algorithms generate explainable models that are represented by an answer set program for classification tasks. Here's a training example for wine dataset:

<code>
	
    model.fit(X_train, Y_train, ratio=0.9)
	
</code>

Note that the hyperparameter **ratio** in **fit** function can be set by the user, and ranges between 0 and 1. Default value is 0.5. This hyperparameter represents the ratio of training examples that are part of the exception to the examples implied by only the default conclusion part of the rule. We recommend that the user experiment with this hyperparameter by trying different values to produce a ruleset with the best F1 score. A range between 0.5 and 0.9 is recommended for experimentation.

The rules generated will be stored in the model object. These rules are organized in a nested intermediate representation. The nested rules will be automatically flattened and decoded to conform to the syntax of answer set programs by calling **print_asp** function: 

<code>
	
    model.print_asp()
	
</code>

An answer set program, compatible with the s(CASP) answer set programming system, is printed as shown below. The s(CASP) system is a system for direclty executing predicate answer set programs in a query-driven manner.

<code>

    % wine dataset (178, 14)
    label(X,'1') :- rule2(X), not rule1(X). 
    label(X,'2') :- rule1(X). 
    label(X,'2') :- rule4(X), not rule1(X), not rule2(X), not rule3(X). 
    label(X,'2') :- rule5(X), not rule1(X), not rule2(X), not rule3(X), not rule4(X). 
    label(X,'3') :- rule3(X), not rule1(X), not rule2(X). 
    rule1(X) :- color_intensity(X,N9), N9=<3.4. 
    rule2(X) :- flavanoids(X,N6), N6>2.03, not ab1(X). 
    rule3(X) :- flavanoids(X,N6), N6=<1.57, not ab2(X). 
    rule4(X) :- alcohol(X,N0), N0>11.56. 
    rule5(X) :- alcohol(X,N0), N0=<11.56. 
    ab1(X) :- proline(X,N12), N12=<678.0. 
    ab2(X) :- hue(X,N10), N10>0.96. 
	
</code>

### Testing in Python
Given **X_test**, a list of test data samples, the Python **predict** function will predict the classification outcome for each of these data samples. 

<code>
	
	Y_test_hat = model.predict(X_test)

</code>

The **classify** function can also be used to classify a single data sample.
	
<code>
	
	y_test_hat = model.classify(x_test)

</code>
	
### Explanation

FOLD-RM provides simple format justification and rebuttal for predictions with **explain** function. 

<code>
	
	model.explain(X_test[i])
	
</code>

Here is a generated justification for an instance prediction in the example above:

<code>
	
    Explanation for example number 1 :
    [F]ab1(X) :- [F]proline(X,N12), N12=<678.0. 
    [T]rule2(X) :- [T]flavanoids(X,N6), N6>2.03, not [F]ab1(X). 
    [F]rule1(X) :- [F]color_intensity(X,N9), N9=<3.4. 
    [T]label(X,'1') :- [T]rule2(X), not [F]rule1(X). 
    {'color_intensity: 6.38', 'proline: 970.0', 'flavanoids: 3.0'}   

</code>

In the generated answers, each literal has been tagged with a label. **[T]** means True, **[F]** means False, and **[U]** means unnecessary to evaluate. And the smallest set of features of the instance is listed for each answer.

## Citations

<code>
	
	@misc{wang2022foldrm,
	      title={FOLD-RM: A Scalable and Efficient Inductive Learning Algorithm for Multi-Category Classification of Mixed Data}, 
	      author={Huaduo Wang and Farhad Shakerin and Gopal Gupta},
	      year={2022},
	      eprint={2202.06913},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}
	}
	
</code>
	
<code>
	
	@misc{wang2021foldr,
	      title={FOLD-R++: A Scalable Toolset for Automated Inductive Learning of Default Theories from Mixed Data}, 
	      author={Huaduo Wang and Gopal Gupta},
	      year={2021},
	      eprint={2110.07843},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}
	}
	
</code>
