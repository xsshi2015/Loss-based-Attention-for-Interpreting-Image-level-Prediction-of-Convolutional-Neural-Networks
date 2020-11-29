import numpy as np
import torch




def calculate_AP(prediction_weights, true_labels):
    _,index = prediction_weights.sort(descending=True)

    n = prediction_weights.shape[0]
    
    ten = (true_labels>0).sum().float()

    p = torch.squeeze(torch.zeros(n,1))
    r = torch.squeeze(torch.zeros(n,1))

    index_set = torch.squeeze(torch.zeros(12,1).long())
    p_max = torch.squeeze(torch.zeros(11,1))

    count = 0.0
    recall_count = 0

    for iter in range(n):
        if true_labels[index[iter]]>0:
            count += 1
            p[iter] = count/(iter+1) 
            r[iter] = count/ten

            if (r[iter]> 0.1*recall_count) and (r[iter]<0.1*(recall_count+1)):
                
                recall_count +=1 
                index_set[recall_count]=iter

            elif r[iter]==1.0:
                recall_count +=1 
                index_set[recall_count]=iter
    

    for iter in range(11):

        if index_set[iter+1]==0:
            p_max[iter] = p[index_set[iter+1]]
        else:
            p_max[iter] = max(p[index_set[iter]:index_set[iter+1]])


    AP = p_max.mean()
    return AP



