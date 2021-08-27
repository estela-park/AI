W = .5
input = .5
lr = .01
epoch = 300

goal = 0.8

for iteration in range(epoch):
    prediction = input*W
    error = prediction - goal
    objective = error **2
    print('objective:', objective, '\tPrediction:',prediction, '\tWeight now:',W)

    pred_to_left = input*(W - lr)
    objective_left = (goal - pred_to_left)**2

    pred_to_right = input*(W + lr)
    objective_right = (goal - pred_to_right)**2

    if objective_right < objective_left:
        print('Weight to right')
        W = W + lr
    elif objective_right > objective_left:
        print('Weight to left')
        W = W - lr
    if prediction==goal:
        print('hit')
        break
