from contextlib import ContextDecorator
import time

class TaskTimer(ContextDecorator):
    """
    A timer that helps trace the execution time
    task_name (str) : name of the task

    >>> Usage
    >>> with TaskTimer("task name"):
    >>>     foo()
    >>>
    >>> @TaskTimer("function name"):
    >>> def foo():
    >>>    pass

    Check the output in logfile or use the eval_task_runtime() function 
    in eval.py to view the average runtime of the task
    """

    def __init__(self, task_name):
        self.task_name = task_name
        #self.tr_logger = TaskRuntimeLogger()

    def __enter__(self):
        self.start_time = time.time()
        print(">>> starting %s" % self.task_name)    
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        print("<<< Complete %s in %s seconds" % (self.task_name, self.interval))
        #self.tr_logger.log(task_name=self.task_name, runtime=self.interval)