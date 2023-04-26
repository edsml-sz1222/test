# 双指针法
# 移除元素

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        fast, slow = 0, 0  # 快慢指针
        size = len(nums)
        while fast < size:  # 快指针遍历数组
            if nums[fast] != val:  # 当快指针指向的元素不等于val时，将其赋值给慢指针指向的元素
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
