import math, re, copy
from scipy.stats import binom

def prune_rules(rules, confidence=0.75):
    # Keep rules with confidence greater than threshold (0.75) and prune others
    pruned_rules = []
    for rule in rules:
        # Check if the rule's confidence value is greater than 0.75
        if rule[-1] >= confidence:
            pruned_rules.append(rule)
    return pruned_rules


def calculate_confidence(tp, total, Z=3):
    z = Z  # Z-score for high confidence
    # Ensure total is not zero to avoid division by zero
    if total == 0:
        return 0.5  # Base confidence when there are no examples
    # Wilson score interval calculation
    p = (tp + (z**2) / 2) / (total + z**2)
    return round(p, 5)




def add_constraint(rules):
    ret, abrules, rx = [], [], []
    k = 1
    for r in rules:
        # Extract confidence if it exists; assume it's the last item
        confidence = r[-1] if isinstance(r[-1], (float, int)) else None
        # Prepare to include confidence in prule if available
        confidence_inclusion = (confidence,) if confidence is not None else tuple()

        if isinstance(r[0], tuple):
            # If confidence is included, modify prule to include it
            prule = (k,) + r[1:-1] + confidence_inclusion # Skip original confidence in r[1:-1]
            # crule remains unchanged as it's more about rule structure than confidence
            crule = (r[0], (k,), tuple([i for i in range(1, k)]))

            ret.append(prule)
            rx.append(crule)
            k += 1
        else:
            # Directly append abstract or simpler rules, including confidence if it exists
            if confidence:
                abrules.append(r[:-1] + (confidence,))
            else:
                abrules.append(r)

    return rx + ret + abrules


def classify(rules, x):
    for rule in rules:
        # Unpack the rule structure.
        classification, conditions, exceptions, confidence = rule
        
        # Evaluate the primary conditions.
        conditions_met = all(evaluate(condition, x) for condition in conditions)
        
        # Check if any exceptions are met. If an exception is met, the rule is invalidated.
        exceptions_met = any(check_exceptions(exception, x) for exception in exceptions)
        
        if conditions_met and not exceptions_met:
            # If conditions are met and no exceptions invalidate the rule, return the classification and confidence.
            return classification[2], confidence  # Assuming classification is a tuple like (-1, '==', 'E')

    return None, None  # If no rules match

def check_exceptions(exception, x):
    """Evaluate whether an exception or its nested exceptions apply to instance x."""
    _, exception_conditions, nested_exceptions, _ = exception
    
    # Check if the primary conditions of the exception are met.
    conditions_met = all(evaluate(condition, x) for condition in exception_conditions)
    
    # For nested exceptions, the logic inverts: if any nested exception is met, the higher exception is invalidated.
    if nested_exceptions:
        nested_exceptions_met = any(check_exceptions(nested_exception, x) for nested_exception in nested_exceptions)
    else:
        nested_exceptions_met = False

    return conditions_met and not nested_exceptions_met


def predict(rules, data):
    ret = []
    for x in data:
        label, confidence = classify(rules, x)  # Capture both label and confidence
        ret.append((label, confidence))  # Append both to the return list
    return ret


def evaluate(item, x):
    def __eval(i, r, v):
        try:
            value = x[i]
        except IndexError:
            print(f"x: {x}")
            print(f"i: {i}")
        if isinstance(v, str):
            if r == '==':
                return x[i] == v
            elif r == '!=':
                return x[i] != v
            else:
                return False
        elif r == '<=':
            try:
                return float(x[i]) <= v
            except ValueError:
                return False  # Handle gaps in the dataset
        elif r == '>':
            try:
                return float(x[i]) > v
            except ValueError:
                return False  # Handle gaps in the dataset
        elif r == '>=':
            try:
                return float(x[i]) >= v
            except ValueError:
                return False  # Handle gaps in the dataset
        elif r == '<':
            try:
                return float(x[i]) < v
            except ValueError:
                return False  # Handle gaps in the dataset
        elif r == '==':
            try:
                return float(x[i]) == v
            except ValueError:
                return False  # Handle gaps in the dataset
        elif r == '!=':
            try:
                return float(x[i]) != v
            except ValueError:
                return False  # Handle gaps in the dataset
        elif isinstance(x[i], str):
            return False
        else:
            return False

    def _eval(i):
        if len(i) == 3:
            return __eval(i[0], i[1], i[2])
        elif len(i) == 4:
            return evaluate(i, x)
    
    if len(item) == 0:
        return False
    if len(item) == 3:
        return __eval(item[0], item[1], item[2])
    if item[3] == 0 and len(item[1]) > 0 and not all([_eval(i) for i in item[1]]):
        return False
    if len(item[2]) > 0 and any([_eval(i) for i in item[2]]):
        return False
    return True






def expand_rules(data, existing_rules, ratio=0.5, improvement_threshold = False, **kwargs):
    ret = []
    current_data = data
    for rule in existing_rules:
        original_rule = copy.deepcopy(rule)
        item = copy.deepcopy(rule[0]) #need to keep the provided literal (target class)
        
        if len(rule) == 4:
            confidence = copy.deepcopy(rule[-1])
        #simplify rule so it works with cover
        
        data_pos, data_neg = split_data_by_item(current_data, item)
        #Evaluate the confidnece of the rule
        
        if len(original_rule) == 3:
            classification, conditions, exceptions = original_rule
        else:
            classification, conditions, exceptions, confidence = original_rule

        #Calculate confidence scores if desired by user
        if original_rule[-1] == 0.5 or len(original_rule) == 3:
            tp = 0
            for d in data_pos:
                conditions_met = all(evaluate(condition, d) for condition in conditions)
                exceptions_met = any(check_exceptions(exception, d) for exception in exceptions)
                if conditions_met and not exceptions_met:
                    tp += 1
            fp = 0
            for d in data_neg:
                conditions_met = all(evaluate(condition, d) for condition in conditions)
                exceptions_met = any(check_exceptions(exception, d) for exception in exceptions)
                if conditions_met and not exceptions_met:
                    fp += 1
            total = tp + fp  # Total = TP + FP

            confidence = calculate_confidence(tp, total)

        rule = (item[0], rule[1], rule[2])
        #Update data to include what is not covered by this rule
        data_fn = []
        for d in data_pos:
                conditions_met = all(evaluate(condition, d) for condition in conditions)
                exceptions_met = any(check_exceptions(exception, d) for exception in exceptions)
                if exceptions_met or not conditions_met:
                    data_fn.append(d)
        data_tn = []
        for d in data_neg:
                conditions_met = all(evaluate(condition, d) for condition in conditions)
                exceptions_met = any(check_exceptions(exception, d) for exception in exceptions)
                if exceptions_met or not conditions_met:
                    data_tn.append(d)
        
        current_data = data_fn + data_tn

        #Restore head of rule and return
        rule = (item, rule[1], rule[2], confidence)
        ret.append(rule)
        
    #Now that the data has been updated and confidence calculated for existing rules if desired we can fit the residuals
    if improvement_threshold == False:        
        ret = ret + foldrm(current_data, ratio=ratio, **kwargs)
    else:
        ret = ret + confidence_foldrm(current_data, improvement_threshold = improvement_threshold, **kwargs)
    return ret


def manual_rule(rule):
    return rule
##############################################


def evaluate_exceptions(rule, data_pos, data_neg, improvement_threshold =0.02):
    #Recursively evaluates and possibly removes exceptions from the rule if they don't affect confidence significantly.
    main_class, main_body, exceptions, original_confidence = rule

    i = 0
    while i < len(exceptions):
        # Recalculate current confidence with the current set of exceptions
        current_rule = (main_class, main_body, exceptions, original_confidence)

        current_tp = len([d for d in data_pos if cover(current_rule, d)])
        current_total = current_tp + len([d for d in data_neg if cover(current_rule, d)])
        current_confidence = calculate_confidence(current_tp, current_total)

        # Temporarily remove one exception for testing
        temp_exceptions = exceptions[:i] + exceptions[i+1:]
        temp_rule = (main_class, main_body, temp_exceptions, original_confidence)
        tp_variant = len([d for d in data_pos if cover(temp_rule, d)])
        total_variant = tp_variant + len([d for d in data_neg if cover(temp_rule, d)])
        variant_confidence = calculate_confidence(tp_variant, total_variant)
        change_in_confidence = current_confidence - variant_confidence

        # Determine if the change is significant
        if change_in_confidence < improvement_threshold:
            #print(f"Removing exception {i} due to insignificant confidence change.")
            exceptions.pop(i)  # Remove the exception permanently
            # Do not increment i, because the list size has been reduced
        else:
            #print(f"Retaining exception {i} due to significant confidence change.")
            if len(exceptions[i][2]) > 0: #check if subrules need to be changed
                evaluate_exceptions(exceptions[i], data_neg, data_pos, improvement_threshold)
            i += 1  # Increment only if the exception is retained


    # Ensure the final rule reflects all changes
    rule = (main_class, main_body, exceptions, original_confidence)
    return rule


def confidence_foldrm(data, improvement_threshold=0.02, ratio=0.5, provided_literal = False, **kwargs):
    ret = []
    
    while len(data) > 0:
        
        if provided_literal == False:
            target_class = most(data) #takes form -1 '==' label
        else:
            target_class = provided_literal

        data_pos, data_neg = split_data_by_item(data, target_class)
        
        rule = learn_confidence_rule(data_pos, data_neg, [], improvement_threshold, ratio, **kwargs)

        if len(rule[2]) >= 1:
            rule = evaluate_exceptions(rule, data_pos, data_neg, improvement_threshold)

        tp = len([d for d in data_pos if cover(rule, d)])  # True positives
        total = tp + len([d for d in data_neg if cover(rule, d)])  # Total = TP + FP
        confidence = calculate_confidence(tp, total)
        
        data_fn = [data_pos[i] for i in range(len(data_pos)) if not cover(rule, data_pos[i])]
        
        if len(data_fn) == len(data_pos):
            break
        data_tn = [data_neg[i] for i in range(len(data_neg)) if not cover(rule, data_neg[i])]
        data = data_fn + data_tn
        rule_with_confidence = tuple([target_class]) + rule[1:-1] + (confidence,) # Attach label and confidence to rule
        
        ret.append(rule_with_confidence) # Append rule with confidence after pruning
    
    return ret

def learn_confidence_rule(data_pos, data_neg, used_items=[], improvement_threshold=0.02, ratio = 0.5, **kwargs):
    items = []
    
    while True:
        t = best_item(data_pos, data_neg, used_items + items, **kwargs)
    
        items.append(t)
        rule = -1, items, [], 0

        
        data_pos = [data_pos[i] for i in range(len(data_pos)) if cover(rule, data_pos[i])]
        data_neg = [data_neg[i] for i in range(len(data_neg)) if cover(rule, data_neg[i])]
        
        if t[0] == -1 or len(data_neg) <= ratio*len(data_pos):
            if t[0] == -1:
                rule = -1, items[:-1], [], 0
            
            if len(data_neg) > 0 and t[0] != -1:
                ab = confidence_fold(data_neg, data_pos, used_items + items, improvement_threshold, **kwargs)
                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break
    return rule


def confidence_fold(data_pos, data_neg, used_items=[], improvement_threshold=0.02, ratio = 0.5):
    ret = []
    while len(data_pos) > 0:
        rule = learn_confidence_rule(data_pos, data_neg, used_items, improvement_threshold, **kwargs)

        data_fn = [data_pos[i] for i in range(len(data_pos)) if not cover(rule, data_pos[i])]
        if len(data_fn) == len(data_pos):
            break
        data_tn = [data_neg[i] for i in range(len(data_neg)) if not cover(rule, data_neg[i])]
        data_pos = data_fn
        data_neg = data_tn
        ret.append(rule)
    return ret


def catch_all_tuples(rule):
    # Define the pattern for the initial part of the rule with an optional confidence prefix
    initial_pattern = r"^(?:with confidence ([0-9]*\.?[0-9]+)\s+)?class\s*=\s*'([^']+)' if '([^']+)' '([^']+)' '([^']+)'"
    
    # Match the initial pattern to ensure rule starts correctly
    initial_match = re.match(initial_pattern, rule)
    if not initial_match:
        raise ValueError("A rule should always begin with an optional \"with confidence #\" followed by \"class = 'label' if 'attribute name/index' 'symbol' 'value'\"")

    # Extract the optional confidence and the initial condition tuple
    confidence = float(initial_match.group(1)) if initial_match.group(1) else None
    found_label = initial_match.group(2)
    initial_tuple = (initial_match.group(2), initial_match.group(3), initial_match.group(4), initial_match.group(5))
    tuple_list = [initial_tuple]
    
    # If confidence is provided, modify the first tuple to include it
    if confidence is not None:
        tuple_list[0] = tuple_list[0] + (confidence,)
    else:
        tuple_list[0] = tuple_list[0] + (0.5,)

    # Remaining part of the string after the initial match
    remaining_string = rule[initial_match.end():]

    # Regex to capture exandors and following conditions
    exandor_pattern = r"\s+(except if|and|or)\s+'([^']+)' '([^']+)' '([^']+)'"
    
    # Search for all occurrences of exandors followed by three arguments
    while remaining_string:
        exandor_match = re.match(exandor_pattern, remaining_string)
        if not exandor_match:
            if remaining_string.strip():  # Check if there's any non-matching remainder
                raise ValueError("Invalid format or missing elements after an exandor")
            break

        # Append the found exandor and the three related values as a tuple
        tuple_list.append((exandor_match.group(1), exandor_match.group(2), exandor_match.group(3), exandor_match.group(4)))
        remaining_string = remaining_string[exandor_match.end():]
    
    return tuple_list

def translate_tuple_to_rule(tuple_list, label_list, attrs, nums):
    
    if tuple_list[0][0] not in label_list:
        raise ValueError(f"The label '{tuple_list[0][0]}' is not a valid class label.")
    
    attr_index = tuple_list[0][1]
    if not isinstance(attr_index, int) or attr_index >= len(attrs):
        if attr_index not in attrs:
            raise ValueError(f"The attribute '{attr_index}' is not valid.")
        attr_index = attrs.index(attr_index)
    
    symbol = tuple_list[0][2]
    if symbol not in ["==", "!=", "<=", "<", ">=", ">"]:
        raise ValueError("Invalid relation symbol.")
    if symbol in ["<=", "<", ">=", ">"]:
        if isinstance(attr_index, int):
            attr = attrs[attr_index]
            if attr not in nums:
                raise ValueError(f"The attribute '{attr}' is not a numeric attribute but is used with a numeric operator '{symbol}'.")
        else:
            if attr_index not in nums:
                raise ValueError(f"The attribute '{attr_index}' is not in the numeric attributes list but is used with a numeric operator '{symbol}'.")

    value = tuple_list[0][3]
    if value.isdigit():  # Check if it's an integer
        new_value = int(value)
    elif is_float(value):  # Check if it's a float
        new_value = float(value)
    else:
        new_value = value  # Keep the original value if it's not a number

    # Update the tuple if the value is a number
    if value.isdigit() or is_float(value):
        tuple_list[0] = (tuple_list[0][0], tuple_list[0][1], tuple_list[0][2], new_value, tuple_list[0][4])
    
    confidence = tuple_list[0][4]
    if not (0 <= confidence <= 1):
        raise ValueError("Confidence must be a numeric value between 0 and 1 inclusive.")
    
    
    rule_list = []
    label_variable = tuple_list[0][0]
    attribute = attr_index
    value = tuple_list[0][3]

    
    # Formulating the first rule list entry
    initial_rule = [((-1, "==", label_variable), [(attribute, symbol, value)], [], confidence)]
    rule_list.append(initial_rule)
    
    
    # Handle subsequent rules if there are any
    previous_rule_was_and = False 
    previous_rule_was_except_if = False
    if len(tuple_list) > 1:
        i = 1
        while i < len(tuple_list):
            current_tuple = tuple_list[i]
            attribute = current_tuple[1]
            if attribute not in attrs:
                raise ValueError(f"The attribute '{attribute}' is not valid.")
            if isinstance(attribute, str):
                attribute = attrs.index(attribute)
    
            symbol = current_tuple[2]
            value = current_tuple[3]

            if value.isdigit():  # Check if it's an integer
                new_value = int(value)
            elif is_float(value):  # Check if it's a float
                new_value = float(value)
            else:
                new_value = value  # Keep the original value if it's not a number
        
            # Update the tuple if the value is a number
            if value.isdigit() or is_float(value):
                current_tuple = (current_tuple[0], current_tuple[1], current_tuple[2], new_value)
            value = new_value

            ### 'Except if' case ###
            if current_tuple[0] == "except if":
                rule_list[-1].append((-1, [(attribute, symbol, value)], [], 0))

            
            ### 'And' case ###
            if current_tuple[0] == "and":
                rule_list[-1][-1][1].append((attribute, symbol, value))
            
            ### 'Or' Case ###
            if current_tuple[0] == "or":
                if len(rule_list[-1][-1][1]) == 1 and len(rule_list[-1]) == 1: #Intended to check if this is a new rule or if it already has and or except if
                    rule_list.append([(rule_list[-1][0], [(attribute, symbol, value)], [], rule_list[-1][0][3])])
                else:
                    if previous_rule_was_and:
                        new_rule = copy.deepcopy(rule_list[-1][-1])
                        new_rule[1].pop()
                        new_rule[1].append((attribute, symbol, value))
                        rule_list.append([new_rule])
                    elif previous_rule_was_except_if:
                        new_rule = copy.deepcopy(rule_list[-1])
                        new_rule.pop()
                        new_rule.append((-1, [(attribute, symbol, value)], [], 0))
                        rule_list.append(new_rule)
                    else:
                        print("A previous entry should have been an and or except if")
            if current_tuple[0] == "and":
                previous_rule_was_and = True
                previous_rule_was_except_if = False
            elif current_tuple[0] == "except if":
                previous_rule_was_and = False
                previous_rule_was_except_if = True
            i += 1
    
    # Handling nested exceptions in the rules
    for rule in rule_list:
        while len(rule) > 1:
            exception = rule.pop()
            rule[-1] = (rule[-1][0], rule[-1][1], [exception], rule[-1][3])
    
    # Flatten the rule list to only contain tuples
    final_rule_list = [rule[0] for rule in rule_list]
    
    return final_rule_list

def append_and_return(lst, element):
    lst.append(element)
    return lst
    
def add_rule(rule, attrs, nums, labels):
    tuple_list = catch_all_tuples(rule)
    rule_list = translate_tuple_to_rule(tuple_list, labels, attrs, nums)
    return rule_list
    
    
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def split_data_by_item(data, item):
    data_pos, data_neg = [], []
    for x in data:
        if evaluate(item, x):
            data_pos.append(x)
        else:
            data_neg.append(x)
    return data_pos, data_neg


def cover(item, x): 
    return evaluate(item, x)


def gain(tp, fn, tn, fp, metric='information_gain', num_classes=2, positive_coverage_weight = 2, beta=1, Z=3):
    formula = metric
    k = num_classes
    if formula == 'information_gain':
        if tp + tn < fp + fn:
            return float('-inf')
        ret = 0
        tot_p, tot_n = float(tp + fp), float(tn + fn)
        tot = float(tot_p + tot_n)
        ret += tp / tot * math.log(tp / tot_p) if tp > 0 else 0
        ret += fp / tot * math.log(fp / tot_p) if fp > 0 else 0
        ret += tn / tot * math.log(tn / tot_n) if tn > 0 else 0
        ret += fn / tot * math.log(fn / tot_n) if fn > 0 else 0
        return ret
    if formula == 'precision':
        if tp ==0:
            return float('-inf')
        return float(tp)/float(tp+fp)
    if formula == 'F1':
        if tp ==0:
            return float('-inf')
        return float(2*tp)/(float(2*tp+fp+fn))
    if formula == 'Jaccard':
        if tp ==0:
            return float('-inf')
        return float(tp)/(float(tp+fp+fn))
    if formula == 'Laplace':
        if tp ==0:
            return float('-inf')
        return float(tp+1)/(float(tp+fp+k))
    if formula == 'Gini_Impurity_Covered':
        if tp ==0:
            return float('-inf')
        return float(tp)/(float(tp+fp)**2) - float(fp)/(float(tp+fp)**2)
    if formula == 'Positive_Coverage_Gain':
        w = positive_coverage_weight 
        if tp ==0:
            return float('-inf')
        PCG = float(tp - w*fp)
        return PCG
    if formula == 'YoudensJ':
        if tp ==0:
            return float('-inf')
        J = float(tp)/float(tp+fn) - float(fp)/float(fp+tn)
        return J
    if formula == 'Weighted_Harmonic_Mean': #Weighted Harmonic Mean of coverage and precision
        if tp ==0:
            return float('-inf')
        Precision = float(tp)/float(tp+fp)
        WHM = (1.0+float(beta)**2) * float(tp) * Precision / (float(beta)**2 * float(tp) + Precision)
        return WHM
    if formula == 'Binomial_Parameter': # Uses Wilson Score interval to choose splits which are statistically significant
        if tp ==0:
            return float('-inf')
        BSP = (float(tp) + float(Z)**2/2)/(float(tp + fp) + float(Z)**2) - float(Z)/(float(tp + fp) + float(Z)**2 ) * sqrt(float(tp*fp)/float(tp+fp) + float(Z)**2/4)
        return BSP
    if formula == 'RPG':
        if tp == 0:
            return float('-inf')
        RPG = float(tp)/float(tp+fp) - float(tp + fn)/float(tp+tn+fp+fn)
        return RPG
    else:
        raise NotImplementedError(f"Metric: '{metric}' is currently not implemented. Instead use one of 'information_gain', 'precision', 'F1', 'Jaccard', 'Laplace', 'Gini_Impurity_Covered', 'Positive_Coverage_Gain', 'YoudensJ', 'Weighted_Harmonic_Mean', 'Binomial_Parameter' or 'RPG'.")
        

metric_list = ['information_gain', 'precision', 'F1', 'Jaccard', 'Laplace', 'Gini_Impurity_Covered', 'Positive_Coverage_Gain', 'YoudensJ', 'Weighted_Harmonic_Mean', 'Binomial_Parameter', 'RPG']

def best_ig(data_pos, data_neg, i, used_items=[], **kwargs):
    xp, xn, cp, cn = 0, 0, 0, 0
    pos, neg = dict(), dict()
    xs, cs = set(), set()
    for d in data_pos:
        if d[i] not in pos:
            pos[d[i]], neg[d[i]] = 0, 0
        pos[d[i]] += 1.0
        if isinstance(d[i], str):
            cs.add(d[i])
            cp += 1.0
        else:
            xs.add(d[i])
            xp += 1.0
    for d in data_neg:
        if d[i] not in neg:
            pos[d[i]], neg[d[i]] = 0, 0
        neg[d[i]] += 1.0
        if isinstance(d[i], str):
            cs.add(d[i])
            cn += 1.0
        else:
            xs.add(d[i])
            xn += 1.0
    xs, cs = list(xs), list(cs)
    xs.sort()
    cs.sort()
    for j in range(1, len(xs)):
        pos[xs[j]] += pos[xs[j - 1]]
        neg[xs[j]] += neg[xs[j - 1]]
    best, v, r = float('-inf'), float('-inf'), ''
    for x in xs:
        if (i, '<=', x) in used_items or (i, '>', x) in used_items:
            continue
        ig = gain(pos[x], xp - pos[x] + cp, xn - neg[x] + cn, neg[x], **kwargs)
        if best < ig:
            best, v, r = ig, x, '<='
        ig = gain(xp - pos[x], pos[x] + cp, neg[x] + cn, xn - neg[x], **kwargs)
        if best < ig:
            best, v, r = ig, x, '>'
    for c in cs:
        if (i, '==', c) in used_items or (i, '!=', c) in used_items:
            continue
        ig = gain(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c], **kwargs)
        if best < ig:
            best, v, r = ig, c, '=='
        ig = gain(cp - pos[c] + xp, pos[c], neg[c], cn - neg[c] + xn, **kwargs)
        if best < ig:
            best, v, r = ig, c, '!='
    return best, r, v


def best_item(X_pos, X_neg, used_items=[], **kwargs):
    ret = -1, '', ''
    if len(X_pos) == 0 and len(X_neg) == 0:
        return ret
    n = len(X_pos[0]) if len(X_pos) > 0 else len(X_neg[0])
    best = float('-inf')
    for i in range(n - 1):
        ig, r, v = best_ig(X_pos, X_neg, i, used_items)
        if best < ig:
            best = ig
            ret = i, r, v #column index, relation, value
    return ret


def most(data, i=-1):
    tab = dict()
    for d in data:  #creating a dictionary of all labels called tab with the number of times that label appears
        if d[i] not in tab: #d[i] is the label
            tab[d[i]] = 0
        tab[d[i]] += 1
    y, n = '', 0
    for t in tab: #find label with most counts
        if n <= tab[t]:
            y, n = t, tab[t]
    return i, '==', y #returns  -1 '==' most common label

def foldrm(data, ratio=0.5, provided_literal = False, **kwargs):
    ret = []
    while len(data) > 0:
        if provided_literal == False: #Determines which class to be evaluated next
            target_class = most(data) #takes form -1 '==' label
        else:
            target_class = provided_literal
        data_pos, data_neg = split_data_by_item(data, target_class) #Splits data into positive and negative examples according to chosen class
        rule = learn_rule(data_pos, data_neg, [], ratio, **kwargs)
        # Calculate confidence here
        tp = len([d for d in data_pos if cover(rule, d)])  # True positives
        total = tp + len([d for d in data_neg if cover(rule, d)])  # Total = TP + FP
        confidence = calculate_confidence(tp, total)
        # Attach label and confidence to rule
        rule_with_confidence = tuple([target_class]) + rule[1:-1] + (confidence,) 
        data_fn = [data_pos[i] for i in range(len(data_pos)) if not cover(rule, data_pos[i])]
        if len(data_fn) == len(data_pos):
            break
        data_tn = [data_neg[i] for i in range(len(data_neg)) if not cover(rule, data_neg[i])]
        data = data_fn + data_tn
        ret.append(rule_with_confidence)  # Append rule with confidence
    return ret

def learn_rule(data_pos, data_neg, used_items=[], ratio=0.5, **kwargs):
    items = []
    while True:
        t = best_item(data_pos, data_neg, used_items + items, **kwargs)
        items.append(t)
        rule = -1, items, [], 0
        data_pos = [data_pos[i] for i in range(len(data_pos)) if cover(rule, data_pos[i])]
        data_neg = [data_neg[i] for i in range(len(data_neg)) if cover(rule, data_neg[i])]
        if t[0] == -1 or len(data_neg) <= len(data_pos) * ratio:
            if t[0] == -1:
                rule = -1, items[:-1], [], 0
            if len(data_neg) > 0 and t[0] != -1:
                ab = fold(data_neg, data_pos, used_items + items, ratio, **kwargs)
                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break
    return rule


def fold(data_pos, data_neg, used_items=[], ratio=0.5, **kwargs):
    ret = []
    while len(data_pos) > 0:
        rule = learn_rule(data_pos, data_neg, used_items, ratio, **kwargs)
        data_fn = [data_pos[i] for i in range(len(data_pos)) if not cover(rule, data_pos[i])]
        if len(data_fn) == len(data_pos):
            break
        data_tn = [data_neg[i] for i in range(len(data_neg)) if not cover(rule, data_neg[i])]
        data_pos = data_fn
        data_neg = data_tn
        ret.append(rule)
    return ret


def flatten_rules(rules):
    abrules = []
    ret = []
    rule_map = dict()
    flatten_rules.ab = -2

    def _eval(i):
        if isinstance(i, tuple) and len(i) == 3:
            return i
        elif isinstance(i, tuple):
            return _func(i)

    def _func(rule, root=False):
        # Assuming confidence intervals are the last two elements of the rule tuple,
        # Extract the rule conditions (excluding the confidence intervals for mapping).
        conditions = rule[:-1]
        confidence = rule[-1]

        t = (tuple(conditions[1]), tuple([_eval(i) for i in conditions[2]])) #A tuple of all conditions in rule including exceptions
        if t not in rule_map:
            rule_map[t] = conditions[0] if root else flatten_rules.ab
            _ret = rule_map[t]
            if root:
                # Include confidence intervals when adding to `ret`.
                ret.append((_ret, t[0], t[1], confidence))
            else:
                # Include confidence intervals when adding to `abrules`.
                abrules.append((_ret, t[0], t[1], confidence))
                flatten_rules.ab -= 1
        elif root:
            # Include confidence intervals when rule is already mapped.
            ret.append((conditions[0], t[0], t[1], confidence))
        return rule_map[t]

    for r in rules:
        _func(r, root=True)
    final_output = ret + abrules
    return final_output

def justify(rs, x, idx=-1, pos=[]):
    for j in range(len(rs)):
        r = rs[j]
        i, d, ab = r[0], r[1], r[2]
        if idx == -1:
            pos.clear()
            if not isinstance(i, tuple):
                continue
            if not isinstance(d[0], tuple):
                if not all([justify(rs, x, idx=_j, pos=pos)[0] for _j in d]):
                    continue
            else:
                if not all([evaluate(_j, x) for _j in d]):
                    continue
        else:
            if i != idx:
                continue
            if not all([evaluate(_j, x) for _j in d]):
                continue
        if len(ab) > 0 and any([justify(rs, x, idx=_j, pos=pos)[0] for _j in ab]):
            continue
        if r not in pos:
            pos.append(r)
        if idx == -1:
            return i[2], j
        else:
            return 1, j
    if idx != -1:
        for r in rs:
            if r[0] == idx and r not in pos:
                pos.append(r)
    if idx == -1:
        return None, -1
    else:
        return 0, -1

