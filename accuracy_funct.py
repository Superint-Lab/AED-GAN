import numpy as np

target_len = 9

# multi step evaluation
def accuracy_ED(y_true, y_pred):

    l1 = [] #target
    l2 = [] #result

    for j in range(len(y_true)):
        l1.append([])
        for i in range(target_len):
            y = np.argmax(y_true[j][i])
            l1[j].append(y)

    for j in range(len(y_pred)):
        l2.append([])
        for i in range(target_len):
            y = np.argmax(y_pred[j][i])
            l2[j].append(y)

    
    # for i in range(1):
    #     print("Target", i, l1[i])
    #     print("Result", i, l2[i])
    #     if l1[i] == l2[i]:
    #         print("Correct")
    #     else:
    #         print("False")

    count = 0
    wronglist = []
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            count += 1
        else:
            wronglist.append(i)

    final_accuracy = count/len(l1)


    step_accuracy_list = []
    count_step = 0
    for step in range(target_len):
        for i in range(len(l1)):
            if l1[i][:step+1] == l2[i][:step+1]:
                count_step += 1

        step_accuracy_list.append(round(100*(count_step/len(l1)), 3))
        count_step = 0

    #print(step_accuracy_list)

    return 100*final_accuracy, step_accuracy_list