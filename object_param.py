class MyClass:
    def __init__(self, init_val):
        self.val = init_val


def main():
    obj = MyClass("first")
    print("before: " + obj.val)
    func(obj)
    print("after: " + obj.val)


def func(obj):
    obj.val = "second"


if __name__ == '__main__':
    # call the main function
    main()
