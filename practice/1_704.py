# 二分查找

class Solution:
    def search_bi(self, nums: List[int], target: int) -> int:  # 左闭右闭
        left = 0
        right = len(nums) - 1  # 闭区间
        while left <= right:  # 区间的合法性判断
            mid = (right + left) // 2  # 可能会越界，可以用mid = left+(right-left)//2
            if nums[mid] > target:
                right = mid - 1  # 闭区间
            elif nums[mid] < target:
                left = mid + 1  # 闭区间
            else:
                return mid
        return -1

    def search_kai(self, nums: List[int], target: int) -> int:  # 左闭右开
        left = 0
        right = len(nums)  # 开区间
        while left < right:  # 区间的合法性判断
            mid = (right + left) // 2  # 可能会越界，可以用mid = left+(right-left)//2
            if nums[mid] > target:
                right = mid  # 开区间
            elif nums[mid] < target:
                left = mid + 1  # 开区间
            else:
                return mid
        return -1
