

class TableDrivenAgent:
    def __init__(self, table: dict, environment: list[str]) -> None:
        self.table = table
        self.environment = environment
        self.percepts = []
        self.actions = []

    def act(self, percept: str) -> str:
        action = self.table[percept] 
        self.percepts.append(percept)
        self.actions.append(action)
        return action

class SimpleReflexAgent:
    """
    Simple Reflex Agent to simulate a Vacuum Cleaner.
    """
    
    def __init__(self, start_location: int, environment: list[str]) -> None:
        self.location = start_location
        self.environment = environment

    def sense(self) -> str:
        return self.environment[self.location]

    def act(self) -> str:
        if self.sense() == "dirty":
            self.environment[self.location] = 'clean'
            self.location += 1
        else:
            self.location += 1

