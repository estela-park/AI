ls = [1, 2, 3, 3, 2, 4, 5]
st = set(ls)
print(st)
print(len(st))

def solution(nums):
    return len(nums) / 2 if (len(nums) / 2) > len(set(nums)) else len(set(nums))

ls1 = [1,2,3,1,2,3,1,2,3]

ls2 = [1,2,3,4,5,6,7,8]

print(solution(ls))
print(solution(ls1))
print(solution(ls2))

answers = [1,3,2,4,2, 1,3,2,4,2, 1,3,2,4,5]
print(sum(1 for i, answer in enumerate(answers) if ((i+1) % 5 == answer) or ( ((i+1) % 5 == 0) and (answer==5)) ))

answers = [0, 0, 0, 3, 0, 4, 0, 5, 0, 0, 2, 0, 2, 0, 0, 0]

print(sum(1 for i, answer in enumerate(answers) if (((i+1) % 2 == 1) and (answer == 2) ) or ( ((i+1) % 8 == 2) and (answer==1)) or ( ((i+1) % 8 == 4) and (answer==3)) or ( ((i+1) % 8 == 6) and (answer==4)) or ( ((i+1) % 8 == 0) and (answer==5)) ))

sum(1 for i, answer in enumerate(answers) if  (((i+1) % 10 == 1) and (answer == 3) )or(((i+1) % 10 == 2) and (answer == 3) )or(((i+1) % 10 == 3) and (answer == 1) )or(((i+1) % 10 == 4) and (answer == 1) )or(((i+1) % 10 == 5) and (answer == 2) )or(((i+1) % 10 == 6) and (answer == 2) )or(((i+1) % 10 == 7) and (answer == 4) )or(((i+1) % 10 == 8) and (answer == 4) )or(((i+1) % 10 == 9) and (answer == 5) )or(((i+1) % 10 == 0) and (answer == 5) ))


