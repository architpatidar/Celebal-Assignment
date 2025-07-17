class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def add_node(self, data):
        new_node = Node(data)
        if not self.head:  
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    def print_list(self):
        if not self.head:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")
    def delete_node(self, n):
        if not self.head:
            print("Cannot delete from an empty list.")
            return
        if n <= 0:
            print("Index should be 1 or greater.")
            return
        if n == 1:
            self.head = self.head.next
            return
        current = self.head
        prev = None
        count = 1
        while current and count < n:
            prev = current
            current = current.next
            count += 1
        if not current:
            print(f"Index {n} is out of range.")
        else:
            prev.next = current.next
if __name__ == "__main__":
    a= LinkedList()
    a.add_node(10)
    a.add_node(20)
    a.add_node(30)
    a.add_node(40)
    print("Initial list:")
    a.print_list()
    a.delete_node(2)
    print("After deleting 2nd node:")
    a.print_list()
    a.delete_node(10)
    empty_a = LinkedList()
    empty_a.delete_node(1)

