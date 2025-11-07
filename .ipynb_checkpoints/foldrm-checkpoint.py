from utils import load_data, split_data, get_scores, justify_data, decode_rules, proof_tree, scores, zip_rule, simplify_rule
from algo import foldrm, predict, classify, flatten_rules, justify, add_constraint, confidence_foldrm, learn_confidence_rule, confidence_fold, expand_rules, add_rule 
import pickle


class Classifier:
    def __init__(self, attrs=None, numeric=None, label=None):
        self.attrs = attrs
        self.numeric = numeric
        self.label = label
        self.rules = None
        self.frs = None
        self.crs = None
        self.asp_rules = None
        self.seq = 1
        self.simple = None
        self.translation = None

    def load_data(self, file, amount=-1):
        if self.label != self.attrs[-1]:
            data, self.attrs = load_data(file, self.attrs, self.label, self.numeric, amount)
        else:
            data, _ = load_data(file, self.attrs[:-1], self.label, self.numeric, amount)
        return data

    def add_manual_rule(self, rule, attrs, nums, labels, instructions=True, FOLD_Syntax = False):
        if self.rules == None:
            self.rules = []
        if instructions:
            print(f"\#\#\#\#\# Instructions \#\#\#\#\#")
            print(f"Manually defined rules should take the following form: \n (with confidence \#) class = 'label' if 'attribute name/index' 'symbol' 'value' ")
            print(f"You can then include additional conditions using 'except if', 'and' or 'or'.")
            print(f"If you wish to use FOLD_Syntax which is useful for people who know it or for copying and pasting rules set FOLD_Syntax = True.")
            print(f"Example Rule 1: with confidence 0.99  class = '0.5' if 'correct_number' '>=' '1' and 'incorrect_unit' '>=' '1' or 'correct_unit' '<=' '0'")
            print(f"Example Rule 2: class = '1' if 'correct_number' '>=' '1' and 'correct_unit' '>=' '1'")
        if FOLD_Syntax:
            self.rules.append(rule)
            print(f"The following rule has been added to the model: {rule}")
        else:
            manual_rules = add_rule(rule, attrs, nums, labels)
            self.rules = self.rules + manual_rules

    def fit(self, data, ratio=0.5, selection_strategy='greedy', **kwargs):
        if self.rules == None:
            self.rules = foldrm(data, ratio=ratio, selection_strategy=selection_strategy)
        elif isinstance(self.rules, list) and len(self.rules)  == 0:
            self.rules = foldrm(data, ratio=ratio, selection_strategy=selection_strategy, **kwargs)
        else:
            self.rules = expand_rules(data, existing_rules = self.rules, ratio=ratio, selection_strategy=selection_strategy, **kwargs)          

    def confidence_fit(self, data, improvement_threshold=0.02, ratio=0.5, selection_strategy='greedy', **kwargs):
        if self.rules == None:
            self.rules = confidence_foldrm(data, improvement_threshold=improvement_threshold, selection_strategy=selection_strategy, **kwargs)
        elif isinstance(self.rules, list) and len(self.rules)  == 0:
            self.rules = confidence_foldrm(data, improvement_threshold=improvement_threshold, selection_strategy=selection_strategy, **kwargs)
        else:
            self.rules = expand_rules(data, existing_rules = self.rules, ratio = 0.5, improvement_threshold = improvement_threshold, selection_strategy=selection_strategy, **kwargs)

    def predict(self, X):
        predictions_with_confidence = predict(self.rules, X)
        return predictions_with_confidence

    def classify(self, x):
        return classify(self.rules, x)
    
    def asp(self, simple=False):
        if (self.asp_rules is None and self.rules is not None) or self.simple != simple:
            self.simple = simple
            self.frs = flatten_rules(self.rules)
            self.frs = self.frs if self.frs is not None else []
            self.frs = [zip_rule(r) for r in self.frs]
            if simple:
                self.asp_rules = decode_rules(self.frs, self.attrs)
            else:
                self.crs = add_constraint(self.frs)
                self.asp_rules = decode_rules(self.crs, self.attrs)
    
            self.asp_rules = self.asp_rules if self.asp_rules is not None else []
            self.crs = self.frs
    
        return self.asp_rules if self.asp_rules is not None else []

    def print_asp(self, simple=False):
        asp_rules_to_print = self.asp(simple=simple)
        for r in asp_rules_to_print:
            print(r)

    def explain(self, x):
        ret = ''
        pos = []
        justify(self.crs, x, pos=pos)  # Assuming justify fills pos with applicable rules
        expl = decode_rules(pos, self.attrs, x=x)  # Decode rules to human-readable format
        for r, e in zip(pos, expl):  # Assume pos and expl are parallel arrays
            confidence = r[-1]  # Extract confidence value
            ret += f"{e} with confidence {confidence}\n"  # Append confidence to explanation
        return ret
    
    def proof(self, x):
        ret = ''
        pos = []
        justify(self.crs, x, pos=pos)  # Assuming justify fills pos with applicable rules
        expl = proof_tree(pos, self.attrs, x=x)  # Generate a proof tree, similar logic to explain
        for r, e in zip(pos, expl):  # Assume pos and expl are parallel arrays
            confidence = r[-1]  # Extract confidence value
            ret += f"{e} with confidence {confidence}\n"  # Append confidence to proof
        return ret


def save_model_to_file(model, file):
    f = open(file, 'wb')
    pickle.dump(model, f)
    f.close()


def load_model_from_file(file):
    f = open(file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret

