def contains_odd(input_list):
    for x in input_list:
        if x % 2 == 1:
            return True
    return False

def is_odd(input_list):
    output_list = []
    for x in input_list:
        if x % 2 == 1:
            output_list.append(True)
        else:
            output_list.append(False)
    return output_list

def element_wise_sum(input_list_1, input_list_2):
    sumList = []
    for x in range(max(len(input_list_1),len(input_list_2))):
        sumList.append(input_list_1[x] + input_list_2[x])
    return sumList

def dict_to_list(input_dict):
    list1 = []
    list1 = input_dict.items()
    return list1
