# basic imports
import numpy as np
import time
import pyqtgraph as pg
import PyQt5 as qt
# framework imports
import cobel.analysis.spatial as sp
                    

class RewardMonitor():
    '''
    Reward monitor. Used for tracking learning progress.
    
    | **Args**
    | trials:                       Maximum number of trials for which the experiment is run.
    | gui_parent:                   The main window for visualization.
    | visual_output:                If true, the learning progress will be plotted.
    | reward_range:                 The range for which the cumulative reward will be plotted.
    '''
    
    def __init__(self, trials, gui_parent, visual_output, reward_range=[0, 1]):
        # store the GUI parent
        self.gui_parent = gui_parent
        # shall visual output be provided?
        self.visual_output = visual_output
        #define the variables that will be monitored
        self.reward_trace = np.zeros(trials, dtype='float')
        self.reward_trace_avg = np.zeros(trials, dtype='float')
        # this is the accumulation range for smoothing the reward curve
        self.avg_window = 20
        
        if self.visual_output:
            # redefine the gui's dimensions
            self.gui_parent.setGeometry(50, 50, 1600, 600)
            # set up the required plots
            self.reward_plot = self.gui_parent.addPlot(title='Trial Reward')
            # set x/y-ranges for the plots
            self.reward_plot.setXRange(0, trials)
            self.reward_plot.setYRange(reward_range[0], reward_range[1])
            # define the episodes domain
            self.trial_domain = np.linspace(0, trials, trials)
            # each variable has a dedicated graph that can be used for displaying the monitored values
            self.reward_trace_graph = self.reward_plot.plot(self.trial_domain, self.reward_trace)
            self.reward_trace_graph_avg = self.reward_plot.plot(self.trial_domain, self.reward_trace_avg)

    def clear_plots(self):
        '''
        This function clears the plots generated by the performance monitor.
        '''
        if self.visual_output:
            self.gui_parent.removeItem(self.reward_plot)
    
    def update(self, logs):
        '''
        This function is called when a trial ends. Here, information about the monitored variables is memorized, and the monitor graphs are updated.
        
        | **Args**
        | logs:                         Information from the reinforcement learning subsystem.
        '''
        # update the reward traces
        rl_reward, trial = logs['trial_reward'], logs['trial']
        self.reward_trace[trial] = rl_reward
        # prepare aggregated reward trace
        self.reward_trace_avg[trial] = np.mean(self.reward_trace[max(0, trial - self.avg_window):(trial + 1)])
        
        if self.visual_output:
            # set the graph's data
            self.reward_trace_graph.setData(self.trial_domain, self.reward_trace, pen=pg.mkPen(color=(128, 128, 128), width=1))
            self.reward_trace_graph_avg.setData(self.trial_domain, self.reward_trace_avg, pen=pg.mkPen(color=(255, 0, 0), width=2))
            # update Qt visuals
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()
            
            
class ResponseMonitor():
    '''
    Response monitor. Used to record responses and display the CRC.
    
    | **Args**
    | trials:                       Maximum number of trials for which the experiment is run.
    | gui_parent:                   The main window for visualization.
    | visual_output:                If true, the learning progress will be plotted.
    '''
    
    def __init__(self, trials, gui_parent, visual_output):
        # store the GUI parent
        self.gui_parent = gui_parent
        # shall visual output be provided?
        self.visual_output = visual_output
        #define the variables that will be monitored
        self.responses = np.zeros(trials, dtype='float')
        self.CRC = np.zeros(trials, dtype='float')
        
        if visual_output:
            # redefine the gui's dimensions
            self.gui_parent.setGeometry(50, 50, 1600, 600)
            # set up the required plots
            self.CRC_plot = self.gui_parent.addPlot(title='CRC')
            # set initial x/y-ranges of the plot
            self.CRC_plot.setXRange(0, 0)
            self.CRC_plot.setYRange(0, 0)
            # define the episodes domain
            self.trial_domain = np.linspace(0, trials, trials)
            # make graph for CRC
            self.CRC_graph = self.CRC_plot.plot(self.trial_domain, self.CRC)

    def clear_plots(self):
        '''
        This function clears the plots generated by the performance monitor.
        '''
        if self.visual_output:
            self.gui_parent.removeItem(self.CRC_plot)
    
    def update(self, logs):
        '''
        This function is called when a trial ends. Here, information about the monitored variables is memorized, and the monitor graphs are updated.
        
        | **Args**
        | logs:                         Information from the reinforcement learning subsystem.
        '''
        trial = logs['trial']
        # store agent's response
        if 'response' in logs:
            self.responses[trial] = logs['response']
        # if no response was defined, code for whether reward was obtained
        else:
            self.responses[trial] = logs['trial_reward'] > 0
        # update CRC
        self.CRC[trial] = np.sum(self.responses[:(trial + 1)])
        
        if self.visual_output:
            # update x/y-ranges of the plot
            self.CRC_plot.setXRange(0, trial)
            self.CRC_plot.setYRange(np.amin(self.CRC), np.amax(self.CRC))
            # set the graph's data
            self.CRC_graph.setData(self.trial_domain, self.CRC, pen=pg.mkPen(color=(128, 128, 128), width=1))
            # update Qt visuals
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()
            
class EscapeLatencyMonitor():
    '''
    Escape latency monitor. Used to record and display the agent's escape latency.
    
    | **Args**
    | trials:                       Maximum number of trials for which the experiment is run.
    | max_steps:                    Maximum number of steps per trial.
    | gui_parent:                   The main window for visualization.
    | visual_output:                If true, the learning progress will be plotted.
    '''
    
    def __init__(self, trials, max_steps, gui_parent, visual_output):
        # store the GUI parent
        self.gui_parent = gui_parent
        self.max_steps = max_steps
        # shall visual output be provided?
        self.visual_output = visual_output
        #define the variables that will be monitored
        self.latency_trace = np.zeros(trials, dtype='float')
        self.latency_trace_avg = np.zeros(trials, dtype='float')
        
        if self.visual_output:
            # redefine the gui's dimensions
            self.gui_parent.setGeometry(50, 50, 1600, 600)
            # set up the required plots
            self.EL_plot = self.gui_parent.addPlot(title='Escape Latency')
            # set initial x/y-ranges of the plot
            self.EL_plot.setXRange(0, trials)
            self.EL_plot.setYRange(0, max_steps * 1.05)
            # define the episodes domain
            self.trial_domain = np.linspace(0, trials, trials)
            # make graph for CRC
            self.EL_graph = self.EL_plot.plot(self.trial_domain, self.latency_trace)
            self.EL_graph_avg = self.EL_plot.plot(self.trial_domain, self.latency_trace_avg)

    def clearPlots(self):
        '''
        This function clears the plots generated by the performance monitor.
        '''
        if self.visual_output:
            self.gui_parent.removeItem(self.EL_plot)
    
    def update(self, logs):
        '''
        This function is called when a trial ends. Here, information about the monitored variables is memorized, and the monitor graphs are updated.
        
        | **Args**
        | logs:                         Information from the reinforcement learning subsystem.
        '''
        trial = logs['trial']
        # update escape latency trace
        self.latency_trace[trial] = logs['steps']
        avg = np.mean(self.latency_trace[max(0, trial-10):(trial+1)])
        self.latency_trace_avg[trial] = self.max_steps if np.isnan(avg) else avg 
        if self.visual_output:
            # set the graph's data
            self.EL_graph.setData(self.trial_domain, self.latency_trace, pen=pg.mkPen(color=(128, 128, 128), width=1))
            self.EL_graph_avg.setData(self.trial_domain, self.latency_trace_avg, pen=pg.mkPen(color=(255, 0, 0), width=2))
            # update Qt visuals
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()
            
            
class RepresentationMonitor():
    '''
    Representation monitor. Used to record and display the agent's internal representation.
    
    | **Args**
    | observations:                 A dictionary containing observations for different input streams.
    | dimensions:                   The spatial dimensions. Required to map observations to spatial locations.
    | model:                        The model used to compute activity maps.
    | layer:                        The layer for which activity maps will be computed.
    | units:                        The units for which activity maps will be stored.
    | update_interval:              The trial interval for which activity maps will be computed.
    | gui_parent:                   The main window for visualization.
    | visual_output:                If true, the learning progress will be plotted.
    '''
    
    def __init__(self, observations, dimensions, model=None, layer=-2, units=np.zeros(1),
                 update_interval=1, gui_parent=None, visual_output=None):
        self.gui_parent = gui_parent
        self.update_interval = update_interval
        self.observations = observations
        self.dimensions = dimensions
        self.model = model
        self.layer = layer
        self.units = units
        self.last_update = 0
        self.process_activity = False
        # shall visual output be provided?
        self.visual_output = visual_output
        # activity trace containing activity maps for specified units across trials (trial, unit, observation)
        self.activity_trace = np.array([sp.get_activity_maps(self.observations, self.model, self.layer, self.units)])
        if self.visual_output:
            # redefine the gui's dimensions
            self.gui_parent.setGeometry(50, 50, 1600, 600)
            # set up the required plots
            self.representation_panel = self.gui_parent.addPlot(title='Activity Map (Unit ' + str(self.units[0]) + ')')
            activity = np.copy(self.activity_trace[0])
            if self.process_activity:
                activity = sp.process_activity_maps(activity)
            current_map = activity[0].reshape(self.dimensions)
            self.activity_map = pg.ImageItem(current_map)
            self.representation_panel.addItem(self.activity_map)
        
    def clear_plots(self):
        '''
        This function clears the plots generated by the performance monitor.
        '''
        if self.visual_output:
            self.gui_parent.removeItem(self.heatmap)
    
    def update(self, logs):
        '''
        This function is called when a trial ends. Here, information about the monitored variables is memorized, and the monitor graphs are updated.
        
        | **Args**
        | trial:                        The actual trial number.
        | logs:                         Information from the reinforcement learning subsystem.
        '''
        trial = logs['trial']
        if (trial - self.last_update) >= self.update_interval:
            self.last_update = trial
            self.activity_trace = np.concatenate((self.activity_trace,
                                                  np.array([sp.get_activity_maps(self.observations, self.model, self.layer, self.units)])))
            if self.visual_output:
                activity = np.copy(self.activity_trace[trial])
                if self.process_activity:
                    activity = sp.process_activity_maps(activity)
                current_map = activity[0].reshape(self.dimensions)
                self.activity_map = pg.ImageItem(current_map)
                self.representation_panel.addItem(self.activity_map)
                # update Qt visuals
                if hasattr(qt.QtGui, 'QApplication'):
                    qt.QtGui.QApplication.instance().processEvents()
                else:
                    qt.QtWidgets.QApplication.instance().processEvents()
            
            
def measure_time_decorator(func):
    '''
    measures the time of an function execution.
    '''
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        value = func(self, *args, **kwargs)
        print(func, "called. elapsed time:", time.perf_counter() - start_time)
        return value

    return wrapper

class UnityPerformanceMonitor:
    def __init__(self, graphicsWindow, visualOutput, reward_plot_viewbox=(-100, 10, 50),
                 steps_plot_viewbox=(0, 1000, 50)):

        # store the rlAgent
        self.graphicsWindow = graphicsWindow

        # shall visual output be provided?
        self.visualOutput = visualOutput

        if visualOutput:
            self.layout = graphicsWindow.centralWidget
            # redefine the gui's dimensions
            self.graphicsWindow.setGeometry(50, 50, 1600, 600)

            # add labels
            self.sps_label = self.layout.addLabel(col=2, justify='center')
            self.sps_label.setFixedHeight(h=10)
            self.nb_episodes_label = self.layout.addLabel("Episode: 0", col=3, justify='center')
            self.nb_episodes_label.setFixedHeight(h=10)
            self.total_steps_label = self.layout.addLabel(col=4, justify='center')
            self.total_steps_label.setFixedHeight(h=10)

            # pens
            self.raw_pen = pg.mkPen((255, 255, 255), width=1)
            self.mean_pen = pg.mkPen((255, 0, 0), width=2)
            self.var_pen = pg.mkPen((0, 255, 0), width=2)

            # viewbox
            self.reward_plot_viewbox = pg.ViewBox(parent=self.layout, enableMouse=True, enableMenu=False)
            self.reward_plot_viewbox.setYRange(min=reward_plot_viewbox[0], max=reward_plot_viewbox[1])
            self.reward_plot_viewbox.setXRange(min=0, max=reward_plot_viewbox[2])
            self.reward_plot_viewbox.setAutoPan(x=True, y=False)
            self.reward_plot_viewbox.enableAutoRange(x=True, y=False)
            self.reward_plot_viewbox.setLimits(xMin=0)

            self.steps_plot_viewbox = pg.ViewBox(parent=self.layout, enableMouse=True, enableMenu=False)
            self.steps_plot_viewbox.setYRange(min=steps_plot_viewbox[0], max=steps_plot_viewbox[1])
            self.steps_plot_viewbox.setXRange(min=0, max=steps_plot_viewbox[2])
            self.steps_plot_viewbox.setAutoPan(x=True, y=False)
            self.steps_plot_viewbox.enableAutoRange(x=True, y=False)
            self.steps_plot_viewbox.setLimits(xMin=0)

            # episode plots
            self.reward_plot_item = self.layout.addPlot(title="reward", viewBox=self.reward_plot_viewbox,
                                                        colspan=3, col=2, row=1)
            self.reward_plot_item.showGrid(x=True, y=True)
            self.reward_graph = self.reward_plot_item.plot()
            self.mean_reward_graph = self.reward_plot_item.plot()

            self.steps_plot_item = self.layout.addPlot(title="steps per episode", viewBox=self.steps_plot_viewbox,
                                                       colspan=3, col=2, row=2)
            self.steps_plot_item.showGrid(x=True, y=True)
            self.steps_graph = self.steps_plot_item.plot()
            self.mean_steps_graph = self.steps_plot_item.plot()

            self.layout.nextRow()

            # the episode range for calculating means and variances
            self.calculation_range = 20

            # data traces
            self.reward_trace = []
            self.nb_episode_steps_trace = []
            self.mean_rewards_trace = []
            self.mean_nb_episode_steps_trace = []
            self.reward_variance_trace = []
            self.nb_episode_steps_variance_trace = []

            # save start time for sps calculation
            self.start_time = time.perf_counter()
            self.sps = 0

    def clearPlots(self):
        '''
        This function clears the plots generated by the performance monitor.
        '''
        if self.visualOutput:
            self.graphicsWindow.removeItem(self.rlRewardPlot)


    def set_episode_data(self, nb_episode, nb_episode_steps, cumulative_reward, nb_step):
        '''
        update the episode data plots
        this is rather slow and is just done on the end of an episode
        :param nb_episode: number of the current episode
        :param nb_episode_steps: the number of steps of this episode
        :param cumulative_reward: the reward for this episode
        :return:
        '''
        # calculate the average steps per second for each episode
        self.sps = int(nb_episode_steps / (time.perf_counter() - self.start_time))
        # reset start time
        self.start_time = time.perf_counter()
        # set the sps label
        self.__set_sps(self.sps)
        # display the total elapsed steps
        self.__set_nb_steps(nb_step)
        self.nb_episodes_label.setText(f'Episode: {nb_episode}')
        self.__set_episode_plot(nb_episode, nb_episode_steps, cumulative_reward)

    def __set_sps(self, sps):
        '''
        set the sps value to the label
        :param sps: steps per second
        :return:
        '''
        self.sps_label.setText("steps per second: " + str(sps))

    def __set_nb_steps(self, steps):
        '''
        set the number of steps to the label
        :param steps: total number of steps
        :return:
        '''
        self.total_steps_label.setText("elapsed steps: " + str(steps))

    def __set_episode_plot(self, nb_episode, nb_episode_steps, episode_reward):
        '''
        calculates the mean and the variance and plots the values in the corresponding graphs.
        :param nb_episode: number of the current episode
        :param nb_episode_steps: the number of steps of this episode
        :param episode_reward: the reward for this episode.
        :return:
        '''
        # append data
        self.reward_trace.append(episode_reward)
        self.nb_episode_steps_trace.append(nb_episode_steps)

        # get the slices for mean calculation
        # if the mean calculation range exceeds the number of gathered value
        # use all existing data as slices
        # else get slices of mean calculation size.
        if nb_episode < self.calculation_range:
            reward_slice = self.reward_trace
            steps_slice = self.nb_episode_steps_trace
        else:
            reward_slice = self.reward_trace[-self.calculation_range:]
            steps_slice = self.nb_episode_steps_trace[-self.calculation_range:]

        # calculate the means
        mean_reward = np.mean(reward_slice)
        mean_steps = np.mean(steps_slice)

        # calculate variances
        var_reward = np.var(reward_slice)
        var_steps = np.var(steps_slice)

        # append the means
        self.mean_rewards_trace.append(mean_reward)
        self.mean_nb_episode_steps_trace.append(mean_steps)

        # append variances
        self.reward_variance_trace.append(var_reward)
        self.nb_episode_steps_variance_trace.append(var_steps)

        # plot series
        self.reward_graph.setData(self.reward_trace, pen=self.raw_pen)
        self.mean_reward_graph.setData(self.mean_rewards_trace, pen=self.mean_pen)

        self.steps_graph.setData(self.nb_episode_steps_trace, pen=self.raw_pen)
        self.mean_steps_graph.setData(self.mean_nb_episode_steps_trace, pen=self.mean_pen)

    def update(self, logs):
        '''
        This function is called when a trial ends. Here, information about the monitored variables is memorized, and the monitor graphs are updated.
    
        trial:  the actual trial number
        logs:   information from the reinforcement learning subsystem
        '''
        # update the reward traces
        cumulative_reward = logs['episode_reward']
        nb_episode_steps = logs['nb_episode_steps']
        elased_steps = logs['nb_steps']

        # update the plots; trial number start from 0
        self.set_episode_data(logs['trial'] + 1, nb_episode_steps, cumulative_reward, elased_steps)

        pg.mkQApp().processEvents()
