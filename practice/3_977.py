# 双指针法
# 有序数组的平方

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums) - 1
        left, right, result_index = 0, n, n  # 左指针，右指针，结果数组指针
        result = [0] * (n + 1)
        while left <= right:
            if nums[left] ** 2 < nums[right] ** 2:
                result[result_index] = nums[right] ** 2
                right -= 1
            else:
                result[result_index] = nums[left] ** 2
                left += 1
            result_index -= 1
        return result
