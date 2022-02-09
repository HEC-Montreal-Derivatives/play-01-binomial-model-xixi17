import numpy as np
from math import log10, exp, sqrt
from scipy.stats import binom
from copy import deepcopy

def generate_state_key_names(state_index):
    """Generator of names for states on a binomial tree.
    
    :param state_index integer: the time index in a binomial model, starting from zero
    :return: state names, list of strings
    :rtype: list

    At index K the generator returns K+1 state names. States are named after the number of Up (U) and Down (D) stock price moves necessary to reach the state.
    """
    state_letters = ['U', 'D']
        
    if state_index == 0:
        return ['State 0']
    else:
        state_digits = int(log10(state_index))+1
        state_pattern_U = "{{0:0{0:d}d}}".format(state_digits)
        state_pattern_D = "{{1:0{0:d}d}}".format(state_digits)
        state_pattern = "State U{0:s}-D{1:s}".format(state_pattern_U, state_pattern_D)
        state_labels = [state_pattern.format(state_index - num_d, num_d) for num_d in range(state_index+1)]
        
        return state_labels

class tree_trunk:
    """Dictionary-based implementation of binomial tree container and a printing method.

    ...

    Attributes
    ----------
    N_steps : int
        number of steps in the tree
    state_names: list containing str
        names of states that will be tracked and printed on the tree
    trunk : dictionary
        core description of the tree
    
    Methods
    -------
    print_leaf(step_index, leaf_index)
        Prints information from a single leaf of the tree, not intended for user interaction
    print(step_indexes)
        Prints the tree at selected time steps
    """
    def __init__(self, N_steps, state_names):
        """
        Parameters
        ----------
        N_steps : int
            number of steps in the tree
        state_names: list containing str
            names of states that will be tracked and printed on the tree
        """
        # Constructor
        
        # Define some self-facts
        self.N_steps = N_steps
        self.state_names = state_names
        
        # Period range
        period_numbers = range(N_steps + 1)
        period_names = ['Period %d' % x for x in period_numbers]
        
        # Initialize dict
        self.trunk = dict.fromkeys(period_names)
        
        # Create state-level dict
        state_dict = dict.fromkeys(state_names)
        
        # Fill states
        trunk_keys = list(self.trunk.keys())
        for pnum in period_numbers:
            state_keys = generate_state_key_names(pnum)
            self.trunk[trunk_keys[pnum]] = dict.fromkeys(state_keys)
            for state_key in self.trunk[trunk_keys[pnum]].keys():
                self.trunk[trunk_keys[pnum]][state_key] = state_dict.copy()
            
    def print_leaf(self, step_index, leaf_index):
        # Printing of information about single leaf
        trunk_keys = list(self.trunk.keys())
        target_key = trunk_keys[step_index]
        branch_keys = list(self.trunk[target_key].keys())
        target_branch_key = branch_keys[leaf_index]
        
        print("In {0:s}".format(target_branch_key))
        for field_key in self.trunk[target_key][target_branch_key].keys():
            field_value = self.trunk[target_key][target_branch_key][field_key]
            if isinstance(field_value, bool):
                print("{0:s} = {1:s}".format(field_key, str(field_value)))
            else:
                print("{0:s} = {1:f}".format(field_key, field_value))
            
    
    def print(self, step_indexes = None):
        """
        Parameters
        ----------
        step_indexes: list of int
            Prints the tree at selected indexes
        """
        # Reasonably pretty print of tree leafs at selected steps

        # If indexes not supplied, print whole tree
        if step_indexes is None:
            step_indexes = list(range(len(self.trunk)))
            print("Printing the entire tree")

        # Check if indexes supplied in a list
        if not isinstance(step_indexes, list):
            print("The time steps you want to print must be supplied as a LIST of INTEGERS,  e.g., [0, 1, 3] (square brackets!)")
            return
        
        # Check if elements of list are integers
        if not all(isinstance(ind, int) for ind in step_indexes):
            print("There are non-integers in your list! Remember, e.g., 1.0 is NOT an INTEGER for python, but 1 IS an INTEGER")
            return

        # Check max
        max_index = max(step_indexes)
        if(max_index > self.N_steps):
            print("You provided a too big state index: {0:d} on a tree of length {1:d}".format(max_index, len(self.trunk)-1))
            return

        # Action
        trunk_keys = list(self.trunk.keys())
        for step_index in step_indexes:
            target_key = trunk_keys[step_index]
            print("--------------------------------------------------------------------")
            print("In Tree At {0:s}:".format(target_key))
            print("--------------------------------------------------------------------")
            for state_idx in range(step_index+1):
                self.print_leaf(step_index, state_idx)
                print("\n")

class binomial_tree:
    """Forward Binomial Tree for pricing Equity Derivatives.

    All math and notation follows the course slides for FINA60211(A) Princpes d'evaluation des produits derives

    Attributes
    ----------
    Stock_0 : float
        initial stock price
    N_steps : int
        number of steps on the tree model
    h_time_step_length : float
        length of a single time step (in years) on the tree: the tree describes stock price behavior over the period of length N_steps * h_time_step_length
    r_int_rate : float
        risk-free interest rate (NOT in percentage pts, i.e., for a rate of 4.2% input 0.042)
    d_div_rate : float
        dividend yield, (NOT in percentage pts, i.e., for a yield of 4.2% input 0.042)
    s_volatility : float
        annualized volatility of the stock return used, alongside other inputs, to calculate the "up" and "down" movement factors
    u_fctr : float
        "up" movement factor calculated from inputs
    d_fctr : float
        "down" movement factor calculated from inputs
    rn_pstar : float
        risk-neutral probability of a single "up" move
    tree : tree_trunk
        object of class tree_trunk containing the binomial tree
    discount_factors : numpy.array
        vector of discount factors from given period to last period
    single_period_discount_factor : float
        exp(- r_int_rate * h_time_step_length)
    

    Methods
    -------
    print(step_indexes)
        Prints the tree at selected time steps
    copy()
        Provides a deep copy of the tree
    pricing(derivative_product)
        Prices a derivative_product (instance of class derivative) and returns it
    """
    def __init__(self, Stock_0, N_steps, h_time_step_length, r_int_rate, d_div_rate, s_volatility):
        """Provide parameters to construct the binomial tree.
        See course slides (Theme 2) for details

        Stock_0 : float
            initial stock price
        N_steps : int
            number of steps on the tree model
        h_time_step_length : float
            length of a single time step (in years) on the tree: the tree describes stock price behavior over the period of length N_steps * h_time_step_length
        r_int_rate : float
            risk-free interest rate (NOT in percentage pts, i.e., for a rate of 4.2% input 0.042)
        d_div_rate : float
            dividend yield, (NOT in percentage pts, i.e., for a yield of 4.2% input 0.042)
        s_volatility : float
            annualized volatility of the stock return used, alongside other inputs, to calculate the "up" and "down" movement factors
        """
        # Save model parameters to self
        self.N_steps = N_steps
        self.h_time_step_length = h_time_step_length
        self.r_int_rate = r_int_rate
        self.d_div_rate = d_div_rate
        self.s_volatility = s_volatility
        
        # Calculate up- and down- factors as in Slide 40 of Deck 2 (forward tree)
        u_fctr = exp((r_int_rate - d_div_rate) * h_time_step_length + s_volatility * sqrt(h_time_step_length))
        d_fctr = exp((r_int_rate - d_div_rate) * h_time_step_length - s_volatility * sqrt(h_time_step_length))
        
        self.u_fctr = u_fctr
        self.d_fctr = d_fctr
        
        # Calculate RN probablity
        rn_probability = exp((r_int_rate - d_div_rate) * h_time_step_length) - d_fctr
        rn_probability = rn_probability / (u_fctr - d_fctr)
        
        self.rn_pstar = rn_probability
        
        # Tree state names
        state_names = ["Stock price", "Probability of State", "\"Up\" transition probability (p-star)", "Multi-Period Discount Factor", "Single-Period Discount Factor", "\"Up\" factor", "\"Down\" factor"]
        
        # Create tree trunk
        self.tree = tree_trunk(N_steps, state_names)
        
        ## Fill tree trunk
        # Multi-Period Discount Factors
        discount_factors = np.linspace(N_steps, 0, N_steps + 1)
        discount_factors = np.exp(- discount_factors * r_int_rate * h_time_step_length)
        
        self.discount_factors = discount_factors
        self.single_period_discount_factor = discount_factors[-2]
        
        time_index = 0
        for time_key in self.tree.trunk.keys():
            loc_df = discount_factors[time_index]
            time_index = time_index + 1
            for state_key in self.tree.trunk[time_key].keys():
                self.tree.trunk[time_key][state_key]["Multi-Period Discount Factor"] = loc_df
                self.tree.trunk[time_key][state_key]["Single-Period Discount Factor"] = discount_factors[-2]
                self.tree.trunk[time_key][state_key]["\"Up\" transition probability (p-star)"] = self.rn_pstar
                self.tree.trunk[time_key][state_key]["\"Up\" factor"] = u_fctr
                self.tree.trunk[time_key][state_key]["\"Down\" factor"] = d_fctr
                
        # Save Multi-Period DFs to DataFrame
        
        # Risk-neutral probabilities of states
        time_index = 0
        for time_key in self.tree.trunk.keys():
            loc_probs = binom.pmf(range(time_index+1), time_index, 1.0 - rn_probability)
            time_index = time_index + 1
            state_index = 0
            for state_key in self.tree.trunk[time_key].keys():
                self.tree.trunk[time_key][state_key]["Probability of State"] = loc_probs[state_index]
                state_index = state_index + 1
                
        # Save RN probabilities to DataFrame
        
        # Stock price values in states
        time_index = 0
        for time_key in self.tree.trunk.keys():
            ud_fctrs = [(u_fctr ** (time_index - num_d)) * (d_fctr ** num_d) for num_d in range(time_index+1)]
            time_index = time_index + 1
            state_index = 0
            for state_key in self.tree.trunk[time_key].keys():
                self.tree.trunk[time_key][state_key]["Stock price"] = Stock_0 * ud_fctrs[state_index]
                state_index = state_index + 1
        
        # Save stock prices to DataFrame
        
    def copy(self):
        return deepcopy(self)

    def print(self, step_indexes = None):
        self.tree.print(step_indexes)
        
    def pricing(self, derivative_product):
        """Price a derivative and return it

        The returned derivative contains a copy of the stock's binomial tree with information about the derivative's valuation added to the tree.

        Parameters
        ----------
        derivative_product : derivative
        :rtype: derivative
        :return: priced derivative product
        """
        
        # Create a copy of own dictionary
        derivative_tree = deepcopy(self.tree)
        
        # Go through the tree in reverse
        pricing_start_periods = list(derivative_tree.trunk.keys())[:-1]
        pricing_start_periods.reverse()
        pricing_end_periods = list(derivative_tree.trunk.keys())[1:]
        pricing_end_periods.reverse()

        # Calculate terminal derivative values
        terminal_period = pricing_end_periods[0]

        for state_index in derivative_tree.trunk[terminal_period].keys():
            stock_price = derivative_tree.trunk[terminal_period][state_index]["Stock price"]
            derivative_payoff = derivative_product.payoff_function(stock_price)
            derivative_tree.trunk[terminal_period][state_index]["Derivative"] = derivative_payoff
            
        ## Recursion up the tree
        # pricing_start_periods is a reversed list of periods, starting at the last-but-one
        # the order of the states in each period is the up-most state at index 0 to the down-most state at index K-1
        time_counter = 0
        for time_index in pricing_start_periods:
            # Get state indices at current time
            state_list = list(derivative_tree.trunk[time_index].keys())

            # Get index of future time -- recall list was reversed, so you pop its first element
            next_time_index = pricing_end_periods.pop(0)

            # Get state indices at future time
            next_time_state_list = list(derivative_tree.trunk[next_time_index].keys())

            # Find values of payoff in states at next time period
            # - For state 0 (upmost), you need 0 and 1, for state 1 you need 1 and 2 etc.
            state_pick = 0
            for state_index in state_list:
                
                # find indices of up and down state
                up_state = next_time_state_list[state_pick]
                down_state = next_time_state_list[state_pick + 1]

                # obtain value of derivative in possible future states
                up_derivative = derivative_tree.trunk[next_time_index][up_state]["Derivative"]
                down_derivative = derivative_tree.trunk[next_time_index][down_state]["Derivative"]

                # one-step ahead binomial pricing
                derivative_value = self.rn_pstar * up_derivative + (1 - self.rn_pstar) * down_derivative
                derivative_value = self.single_period_discount_factor * derivative_value

                # If American option, check if intrinsic value greater than discounted expected value
                if derivative_product.exercise_type == "American":
                    current_stock_price = derivative_tree.trunk[time_index][state_index]["Stock price"]
                    intrinsic_value = derivative_product.payoff_function(current_stock_price)
                    if intrinsic_value > derivative_value:
                        derivative_value = intrinsic_value
                        early_exercise = True
                    else:
                        early_exercise = False
                
                # Write value to tree
                derivative_tree.trunk[time_index][state_index]["Derivative"] = derivative_value
                if derivative_product.exercise_type == "American":
                    derivative_tree.trunk[time_index][state_index]["Exercise Derivative"] = early_exercise

                # replicating portfolio: repl_delta and B
                current_stock_price = derivative_tree.trunk[time_index][state_index]["Stock price"]
                repl_delta = up_derivative - down_derivative
                repl_delta = repl_delta / (current_stock_price * self.u_fctr - current_stock_price * self.d_fctr)
                repl_delta = repl_delta * np.exp(-self.d_div_rate * self.h_time_step_length)

                repl_b = self.u_fctr * down_derivative - self.d_fctr * up_derivative
                repl_b = repl_b / (self.u_fctr - self.d_fctr)
                repl_b = repl_b * self.single_period_discount_factor

                if derivative_product.exercise_type == "European":
                    derivative_tree.trunk[time_index][state_index]["Repl. Delta"] = repl_delta
                    derivative_tree.trunk[time_index][state_index]["Repl. Bond"] = repl_b

                # Update state selector
                state_pick = state_pick + 1

        derivative_product.set_tree(derivative_tree)
        return derivative_product

#### Define derivative classes ####

class derivative:
    """Generic derivative class

    Attributes
    ----------
    payoff_function : function
        function used to calculate payoff values
    exercise_type : str
        "European" or "American"
    pricing_tree : tree_trunk
        Initally None, set after calling binomial_tree.pricing() on the derivative
    
    Methods
    -------
    set_tree(tree_trunk)
        Sets the pricing_tree object
    print(step_indexes)
        Prints the tree at selected time steps
    """
    def __init__(self, payoff_function, exercise_type = "European"):
        self.payoff_function = payoff_function
        self.exercise_type = exercise_type
        self.pricing_tree = None

    def set_tree(self, binomial_tree):
        self.pricing_tree = binomial_tree
    
    def print(self, step_indexes = None):
        if self.pricing_tree is not None:
            self.pricing_tree.print(step_indexes)
        else:
            print("The derivative has not been priced yet")
            return

## Helper functions
def call_payoff(strike_price, terminal_stock_price):
    """Terminal payof of a call option
    max(S - K, 0)
    """
    return np.maximum(terminal_stock_price - strike_price, 0.0)

def put_payoff(strike_price, terminal_stock_price):
    """Terminal payof of a put option
    max(K - S, 0)
    """
    return np.maximum(strike_price - terminal_stock_price, 0.0)

class european_call(derivative):
    """
    European call
    """
    def __init__(self, strike_price):
        """Define an European Call

        Parameters
        ----------
        strike_price : float
            strike price of the option
        """
        self.strike_price = strike_price

        def loc_payoff(terminal_stock_price):
            return call_payoff(self.strike_price, terminal_stock_price)

        derivative.__init__(self, loc_payoff, "European")

class american_call(derivative):
    """
    American call
    """
    def __init__(self, strike_price):
        """Define an American Call

        Parameters
        ----------
        strike_price : float
            strike price of the option
        """
        self.strike_price = strike_price

        def loc_payoff(terminal_stock_price):
            return call_payoff(self.strike_price, terminal_stock_price)

        derivative.__init__(self, loc_payoff, "American")

class european_put(derivative):
    """
    European put
    """
    def __init__(self, strike_price):
        """Define an European Put

        Parameters
        ----------
        strike_price : float
            strike price of the option
        """

        self.strike_price = strike_price

        def loc_payoff(terminal_stock_price):
            return put_payoff(self.strike_price, terminal_stock_price)

        derivative.__init__(self, loc_payoff, "European")

class american_put(derivative):
    """
    American put
    """
    def __init__(self, strike_price):
        """Define an American Put

        Parameters
        ----------
        strike_price : float
            strike price of the option
        """

        self.strike_price = strike_price

        def loc_payoff(terminal_stock_price):
            return put_payoff(self.strike_price, terminal_stock_price)

        derivative.__init__(self, loc_payoff, "American")