def solution(nums):
    return [f'{i}, {k}, {m}' for i, j in enumerate(nums) for k, l in enumerate(nums[i+1:]) for m, n in enumerate(nums[k+1:])]

nums_input = [1, 2, 3, 4, 5, 6]

print(solution(nums_input))

nums2 = [1, 5, 2 ,3 , 4]

for i, j in enumerate(nums2):
    for k, l in enumerate(nums2[i+1:]):
        for m, n in enumerate(nums2[nums2.index(l)+1:]):
            print(j,l,n,(sum(1 for o in range(2, j+l+n) if (j+l+n) % o == 0) == 0))
