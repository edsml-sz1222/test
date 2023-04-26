# 滑动窗口
# 双指针

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        start, sums = 0, 0
        length = len(nums)
        min_len = length + 1  # 初始最小长度，可以选择无穷大，但只要比数组长我认为就可以
        for end in range(length):
            sums += nums[end]
            while sums >= target:  # start指针右移，直到总和小于target
                min_len = min(min_len, end - start + 1)  # 更新最小长度
                sums -= nums[start]
                start += 1
        if min_len == length + 1:   # 如果没有找到，返回0
            return 0
        else:
            return min_len
