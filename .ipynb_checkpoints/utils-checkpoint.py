import random, re
from algo import evaluate, justify, prune_rules
from statistics import mean, stdev

def get_inverse_brier_score(Ystar_tuples, Y_test):
    #Note Ystar_tuples should be a list of tuples of length 2. The first entry is the predicted class, the second entry is the confidence.
    #Note Ystar should be a list of strings (classes)
    
    #First check that the lengths of the two lists are the same:
    if len(Ystar_tuples) != len(Y_test):
        return 0
    
    num_examples = len(Y_test)
    Cumulative_Brier = float(0)
    for i in range(num_examples):
        if Ystar_tuples[i][0] == None:
            Cumulative_Brier += 0.25
        else:
            if Ystar_tuples[i][0] == Y_test[i]:
                Cumulative_Brier += (1-Ystar_tuples[i][1])**2
            else:
                Cumulative_Brier += (Ystar_tuples[i][1])**2
    Brier_score = Cumulative_Brier/num_examples
    Inverse_Brier_score = 1 - Brier_score
    return Inverse_Brier_score







def split_xy(data):
    feature = []
    label = []
    for d in data:
        feature.append(d[:-1])  # Extract features as before
        label.append(d[-1])  # Append the label as is, without conversion
    return feature, label

###This version of the function allows for predictable splitting which is useful for testing. ###
"""def split_data(data, ratio=0.8):
    num = int(len(data) * ratio)
    
    # Split the data
    train = data[:num]
    test = data[num:]
    
    return train, test"""


def run_trials(data, model, num_trials=30):
    scores = []
    rule_counts = []  # To store the rule count for each trial
    model.rules = []
    predicate_counts = []
    for i in range(num_trials):
        data_train, data_test = split_data(data, ratio=0.8)  # data splitting function
        X_test, Y_test = split_xy(data_test)
        model.asp_rules=None
        model.rules=None
        model.fit(data_train, ratio=0.5)  # Train model on training set
        Ystar_test_tuples = model.predict(X_test)  # Predict on test set
        Ystar_test = [y[0] for y in Ystar_test_tuples]  # Extract predicted labels
        
        score = get_scores(Ystar_test, data_test)  # Calculate score for this trial
        scores.append(score)

        rule_count = count_rules_in_model(model)  # Use the function to count rules in the model
        rule_counts.append(rule_count)  # Append rule count for this trial

        model.asp()
        predicate_count = num_predicates(model)
        #print(f"Trial {i+1}: Predicate count = {predicate_count}")
        predicate_counts.append(predicate_count)

    # Calculate average and standard deviation for scores and rule counts
    average_score = mean(scores)
    score_stdev = stdev(scores) if len(scores) >= 1 else 0

    average_rule_count = mean(rule_counts)
    rule_count_stdev = stdev(rule_counts) if len(rule_counts) >= 1 else 0

    average_predicate_count = mean(predicate_counts)
    predicate_count_stdev = stdev(predicate_counts) if len(predicate_counts) >= 1 else 0

    return average_score, score_stdev, average_rule_count, rule_count_stdev, average_predicate_count, predicate_count_stdev


def run_pruned_trials(data, model, num_trials=30, confidence_threshold=0.75):
    scores = []
    rule_counts = []  # To store the rule count for each trial
    predicate_counts = []
    for _ in range(num_trials):
        data_train, data_test = split_data(data, ratio=0.8)  # Split data into training and testing sets
        X_test, Y_test = split_xy(data_test)
        model.asp_rules=None
        model.rules=None
        model.fit(data_train, ratio=0.5)  # Train model on training set

        # Prune rules with confidence < threshold
        model.rules = prune_rules(model.rules, confidence_threshold)

        Ystar_test_tuples = model.predict(X_test)  # Predict on test set
        Ystar_test = [y[0] for y in Ystar_test_tuples]  # Extract predicted labels
        
        score = get_scores(Ystar_test, data_test)  # Calculate score for this trial
        scores.append(score)

        rule_count = count_rules_in_model(model)  # Count rules in the pruned model
        rule_counts.append(rule_count)  # Append rule count for this trial

        model.asp()
        predicate_count = num_predicates(model)
        #print(f"Trial {i+1}: Predicate count = {predicate_count}")
        predicate_counts.append(predicate_count)

    # Calculate average and standard deviation for scores and rule counts
    average_score = mean(scores)
    score_stdev = stdev(scores) if len(scores) >= 1 else 0

    average_rule_count = mean(rule_counts)
    rule_count_stdev = stdev(rule_counts) if len(rule_counts) >= 1 else 0

    average_predicate_count = mean(predicate_counts)
    predicate_count_stdev = stdev(predicate_counts) if len(predicate_counts) >= 1 else 0

    return average_score, score_stdev, average_rule_count, rule_count_stdev, average_predicate_count, predicate_count_stdev




def run_improved_pruned_trials(data, model, num_trials=30, confidence_threshold=0.75, improvement_threshold=0.02):
    scores = []
    rule_counts = []  # To store the rule count for each trial
    predicate_counts = []
    for i in range(num_trials):
        data_train, data_test = split_data(data, ratio=0.8)  # Split data into training and testing sets
        X_test, Y_test = split_xy(data_test)
        #print(f"confidence threshold: {confidence_threshold}, improvment threshold: {improvement_threshold}")
        model.asp_rules=None
        model.rules=None
        model.confidence_fit(data_train, improvement_threshold=improvement_threshold)  # Train model on training set
        # Prune rules with confidence < threshold
        model.rules = prune_rules(model.rules, confidence_threshold)
        
        Ystar_test_tuples = model.predict(X_test)  # Predict on test set
        Ystar_test = [y[0] for y in Ystar_test_tuples]  # Extract predicted labels
        
        score = get_scores(Ystar_test, data_test)  # Calculate score for this trial
        scores.append(score)

        rule_count = count_rules_in_model(model)  # Count rules in the pruned model
        rule_counts.append(rule_count)  # Append rule count for this trial

        model.asp()
        predicate_count = num_predicates(model)
        #print(f"Trial {i+1}: Predicate count = {predicate_count}")
        predicate_counts.append(predicate_count)

    # Calculate average and standard deviation for scores and rule counts
    average_score = mean(scores)
    score_stdev = stdev(scores) if len(scores) >= 1 else 0

    average_rule_count = mean(rule_counts)
    rule_count_stdev = stdev(rule_counts) if len(rule_counts) >= 1 else 0

    average_predicate_count = mean(predicate_counts)
    predicate_count_stdev = stdev(predicate_counts) if len(predicate_counts) >= 1 else 0

    return average_score, score_stdev, average_rule_count, rule_count_stdev, average_predicate_count, predicate_count_stdev


def count_rules_in_rule(rule):
    # Start with 1 for the current rule
    count = 1
    # If the rule has exceptions
    if len(rule) > 2 and rule[2]:
        for exception in rule[2]:
            # Count each exception and if the exception has its own exceptions, count those recursively
            count += count_rules_in_rule(exception)
    return count

def count_rules_in_model(model):
    total_rules = 0
    if model.rules is not None:
        for rule in model.rules:
            total_rules += count_rules_in_rule(rule)
    return total_rules


def load_data(file, attrs, label, numerics, amount=-1):
    f = open(file, 'r')
    attr_idx, num_idx, lab_idx = [], [], -1
    ret, i, k = [], 0, 0
    head = ''
    for line in f.readlines():
        if i == 0:
            line = line.strip('\n').split(',')
            attr_idx = [j for j in range(len(line)) if line[j] in attrs]
            num_idx = [j for j in range(len(line)) if line[j] in numerics]
            for j in range(len(line)):
                if line[j] == label:
                    lab_idx = j
                    head += line[j]
        else:
            line = line.strip('\n').split(',')
            r = [j for j in range(len(line))]
            for j in range(len(line)):
                if j in num_idx:
                    try:
                        r[j] = float(line[j])
                    except:
                        r[j] = line[j]
                else:
                    r[j] = line[j]
            r = [r[j] for j in attr_idx]
            if lab_idx != -1:
                y = line[lab_idx]
                r.append(y)
            ret.append(r)
        i += 1
        amount -= 1
        if amount == 0:
            break
    attrs.append(head)
    return ret, attrs


def split_data(data, ratio=0.8, shuffle=True):
    if shuffle:
        random.shuffle(data)
    num = int(len(data) * ratio)
    train, test = data[: num], data[num:]
    return train, test

def split_data_stratified(data, ratio=0.8, shuffle=True):
    if shuffle:
        random.shuffle(data)  # Randomly shuffle the data if shuffle is True

    # Extract labels to stratify by them
    label_indices = {}
    for index, row in enumerate(data):
        label = row[-1]  # Assuming the label is the last element in each row
        if label in label_indices:
            label_indices[label].append(index)
        else:
            label_indices[label] = [index]

    # Initialize lists for training and testing indices
    train_indices = []
    test_indices = []

    # Distribute indices to ensure at least one example of each label in both sets
    for indices in label_indices.values():
        split_point = max(1, int(len(indices) * ratio))  # Ensure at least one sample goes to test
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    # Convert indices back to actual data
    data_train = [data[i] for i in train_indices]
    data_test = [data[i] for i in test_indices]

    return data_train, data_test

def over_sample(data, each=-1):
    tab = {}
    for d in data:
        y = d[-1]
        if y not in tab:
            tab[y] = []
        tab[y].append(d)
    ret, n = [], 0
    for t in tab:
        n = max(n, len(tab[t]))
    n = each if each > 0 else n
    for t in tab:
        d = tab[t] * int(n / len(tab[t]) + 1)
        random.shuffle(d)
        ret.extend(d[:n])
    tmp = [d[-1] for d in ret]
    tab = {}
    for t in tmp:
        if t not in tab:
            tab[t] = 0
        tab[t] += 1
    print('% over sample size', len(ret), tab)
    return ret


def get_scores(Y_hat, data):
    n = len(Y_hat)
    #print(f"n={n}")
    m = 0
    for i in range(n):
        if Y_hat[i] == data[i][-1]:
            m += 1
    return float(m) / n


def scores(Y_hat, Y, weighted=False):
    n = len(Y_hat)
    tp, fp, fn = {}, {}, {}
    for i in range(n):
        y, yh = Y[i], Y_hat[i]
        if y not in tp:
            tp[y], fp[y], fn[y] = 0, 0, 0
        if yh not in tp:
            tp[yh], fp[yh], fn[yh] = 0, 0, 0
        if yh == y:
            tp[y] += 1
        else:
            fp[yh] += 1
            fn[y] += 1
    p_mic = float(sum([tp[y] for y in tp])) / sum([tp[y] + fp[y] for y in tp])
    if weighted:
        p_mac = 1.0 / n * sum([float(tp[y]) * (tp[y] + fn[y]) / (tp[y] + fp[y]) for y in tp if tp[y] + fp[y] > 0])
        r_mac = 1.0 / n * sum([float(tp[y]) * (tp[y] + fn[y]) / (tp[y] + fn[y]) for y in tp if tp[y] + fn[y] > 0])
        f1_mac = 2.0 / n * sum([(tp[y] + fn[y]) * (float(tp[y]) / (tp[y] + fp[y]) * float(tp[y]) / (tp[y] + fn[y]))
                                / (float(tp[y]) / (tp[y] + fp[y]) + float(tp[y]) / (tp[y] + fn[y])) for y in tp
                                if tp[y] + fn[y] > 0 and tp[y] + fp[y] > 0 and tp[y] > 0])
    else:
        p_mac = 1.0 / len(tp) * sum([float(tp[y]) / (tp[y] + fp[y]) for y in tp if tp[y] + fp[y] > 0])
        r_mac = 1.0 / len(tp) * sum([float(tp[y]) / (tp[y] + fn[y]) for y in tp if tp[y] + fn[y] > 0])
        f1_mac = 2 * (p_mac * r_mac) / (p_mac + r_mac) if p_mac + r_mac > 0 else 0
    return p_mic, p_mac, r_mac, f1_mac


def justify_data(frs, x, attrs):
    ret = []
    for r in frs:
        d = r[1]
        if isinstance(d[0], tuple):
            for j in d:
                ret.append(attrs[j[0]] + ': ' + str(x[j[0]]))
    return set(ret)


def decode_rules(rules, attrs, x=None):
    ret = []
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}

    def _f1(it):
        prefix, not_prefix = '', ''
        if isinstance(it, tuple) and len(it) == 3:
            if x is not None:
                if it[0] == -1:
                    prefix = '[T]' if justify(rules, x)[0] == it[2] else '[F]'
                else:
                    prefix = '[T]' if evaluate(it, x) else '[F]'
                not_prefix = '[T]' if prefix == '[F]' else '[F]'
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -i - 2
                r = nr[r]
            k = attrs[i].lower().replace(' ', '_')
            if isinstance(v, str):
                v = v.lower().replace(' ', '_')
                v = 'null' if len(v) == 0 else '\'' + v + '\''
            if r == '==':
                return prefix + k + '(X,' + v + ')'
            elif r == '!=':
                return 'not ' + not_prefix + k + '(X,' + v + ')'
            else:
                return prefix + k + '(X,' + 'N' + str(i) + ')' + ', N' + str(i) + r + str(round(v, 5))
        elif it == -1:
            pass
        else:
            if x is not None:
                if it not in [r[0] for r in rules]:
                    prefix = '[U]'
                else:
                    prefix = '[T]' if justify(rules, x, it)[0] else '[F]'
                    pass
            if it < -1:
                return prefix + 'ab' + str(abs(it) - 1) + '(X)'
            else:
                return prefix + 'rule' + str(abs(it)) + '(X)'
    
    def _f2(rule):
        head = _f1(rule[0])
        body_elements = [ _f1(i) for i in list(rule[1]) ]
        tail_elements = [ _f1(i) for i in list(rule[2]) ]
        
        # Ensure body elements are concatenated with commas
        body = ', '.join(body_elements)
        # Ensure tail elements are concatenated with ', not ' if they are present
        tail = ', not '.join(tail_elements) if tail_elements else ''
        # Add 'not ' prefix to the tail if it's not empty
        tail = 'not ' + tail if tail else tail
    
        # Concatenate head, body, and tail with proper syntax
        rule_str = f"{head} :- {body}" + (f", {tail}" if tail else "") + "."
    
        # Append confidence if present
        confidence = rule[-1] if isinstance(rule[-1], (float, int)) else None
        confidence_str = f" [confidence: {round(confidence, 5)}]" if confidence else ""
        rule_str += confidence_str
    
        return rule_str

    for _r in rules:
        ret.append(_f2(_r))
    return ret
def zip_rule(rule):
    tab, dft = {}, []
    # Extract the confidence value from the rule
    confidence = rule[-1]  # Confidence is the last element of the rule tuple

    for i in rule[1]:  # Iterate over the conditions
        if isinstance(i[2], str):  # Handle categorical attributes
            dft.append(i)
        else:  # Handle numeric attributes
            if i[0] not in tab:
                tab[i[0]] = []
            if i[1] == '<=':
                tab[i[0]].append([float('-inf'), i[2]])
            else:  # Assume '>' operator
                tab[i[0]].append([i[2], float('inf')])

    nums = [t for t in tab]
    nums.sort()
    for i in nums:
        left, right = float('inf'), float('-inf')
        for j in tab[i]:
            if j[0] == float('-inf'):
                left = min(left, j[1])
            else:
                right = max(right, j[0])
        if left == float('inf'):
            dft.append((i, '>', right))
        elif right == float('-inf'):
            dft.append((i, '<=', left))
        else:
            dft.append((i, '>', right))
            dft.append((i, '<=', left))
    # Return the updated rule structure, now including the confidence value
    transformed_rule = rule[0], dft, rule[2], confidence
    return rule[0], dft, rule[2], confidence

def simplify_rule(rule):
    head, body = rule.split(' :- ')
    items = body.split(', ')
    items = list(dict.fromkeys(items))
    body = ', '.join(items)
    return head + ' :- ' + body


def num_predicates(model):
    predicates = set()
    
    for rule in model.asp_rules:
        # Remove confidence values
        rule = re.sub(r'\[confidence:.*?\]', '', rule).strip()
        
        # Split the rule into parts
        parts = rule.split(':-')
        if len(parts) > 1:
            body = parts[1].strip()
            
            # Split the body into individual predicates
            body_predicates = re.split(r',\s*(?=\w+\()', body)
            
            for pred in body_predicates:
                # Remove leading 'not'
                pred = re.sub(r'^not\s+', '', pred)
                
                # Extract the main predicate and its conditions
                match = re.match(r'(\w+\([^)]+\))(.*)$', pred)
                if match:
                    main_pred, conditions = match.groups()
                    if conditions:
                        conditions = conditions.strip(', ')
                        if conditions and not conditions.startswith('not'):
                            predicates.add(f"{main_pred}, {conditions}")
                    else:
                        predicates.add(main_pred)
    
    # Remove rule predicates and clean up remaining predicates
    cleaned_predicates = set()
    for pred in predicates:
        if not pred.startswith('rule'):
            # Remove 'not ab1(X)' and 'not ab2(X)' parts
            cleaned_pred = re.sub(r',\s*not\s+ab[12]\(X\)', '', pred)
            cleaned_predicates.add(cleaned_pred)
    
    # Print the list of predicates
    #print("List of predicates:")
    #for pred in sorted(cleaned_predicates):
    #    print(f"- {pred}")
    
    return len(cleaned_predicates)


def fitem(rules, attrs, x, it):
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}
    if isinstance(it, tuple) and len(it) == 3 and not isinstance(it[2], tuple) and it[0] != -1:
        suffix = ' (DOES HOLD) ' if evaluate(it, x) else ' (DOES NOT HOLD) '
        i, r, v = it[0], it[1], it[2]
        if i < -1:
            i = -2 - i
            r = nr[r]
        k = attrs[i].lower().replace(' ', '_')
        if isinstance(v, str):
            v = v.lower().replace(' ', '_')
            v = 'null' if len(v) == 0 else '\'' + v + '\''
        xi = x[i]
        if isinstance(xi, str):
            xi = xi.lower().replace(' ', '_')
            xi = '\'null\'' if len(xi) == 0 else xi
        if r == '==':
            return 'the value of ' + k + ' is \'' + str(xi) + '\' which should equal ' + v + suffix
        elif r == '!=':
            return 'the value of ' + k + ' is \'' + str(xi) + '\' which should not equal ' + v + suffix
        else:
            if r == '<=':
                return 'the value of ' + k + ' is ' + str(xi) + ' which should be less equal to ' + str(round(v, 5)) + suffix
            else:
                return 'the value of ' + k + ' is ' + str(xi) + ' which should be greater than ' + str(round(v, 5)) + suffix
    elif isinstance(it, tuple) and len(it) == 3 and not isinstance(it[2], tuple) and it[0] == -1:
        suffix = ' DOES HOLD ' if justify(rules, x)[0] else ' DOES NOT HOLD '
        return 'the value of ' + attrs[-1] + ' is ' + str(it[2]) + suffix
    else:
        if it not in [r[0] for r in rules]:
            pass
        elif it < -1:
            suffix = ' DOES HOLD ' if justify(rules, x, it)[0] else ' DOES NOT HOLD '
            return 'exception ab' + str(abs(it) - 1) + suffix
        else:
            suffix = ' DOES HOLD ' if justify(rules, x, it)[0] else ' DOES NOT HOLD '
            return 'rule' + str(it) + suffix


def frules(rules, attrs, x, rule, indent=0):
    head = '\t' * indent + fitem(rules, attrs, x, rule[0]) + 'because \n'
    body = ''
    if not isinstance(rule[0], tuple):
        for i in list(rule[1]):
            body = body + '\t' * (indent + 1) + fitem(rules, attrs, x, i) + '\n'
    else:
        for i in list(rule[1]):
            if isinstance(i, tuple):
                body = body + '\t' * (indent + 1) + fitem(rules, attrs, x, i) + '\n'
            else:
                for r in rules:
                    if i == r[0]:
                        body = body + frules(rules, attrs, x, r, indent + 1)
    tail = ''
    for i in list(rule[2]):
        for r in rules:
            if i == r[0]:
                tail = tail + frules(rules, attrs, x, r, indent + 1)
    _ret = head + body + tail
    chars = list(_ret)
    _ret = ''.join(chars)
    return _ret


def proof_tree(rules, attrs, x):
    ret = []
    for r in rules:
        if isinstance(r[0], tuple):
            ret.append(frules(rules, attrs, x, r))
    return ret
