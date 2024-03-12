# make a workflow saver that saves the workflow in a file and can load the workflow from the file and be used again

# Create a class Workflow
class Workflow:
    def __init__(self, filename):
        self.filename = filename

    def save(self, workflow):
        with open (self.filename, 'w') as file:
            file.write (str (workflow))

    def load(self):
        with open (self.filename, 'r') as file:
            return file.read ()


workflow = Workflow ('workflow.txt')
workflow.save ('This is a workflow')
print (workflow.load ())
