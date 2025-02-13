class TimeSeries:
    """Class for time series of statistics during training

    Attributes
    ----------
    name (str): the name of the time series
    _statistic (Statistic): the underlying statistic whose value is measured regularly to define the time series
    _time_series (list): the time series. Each entry corresponds to the value of the statistic at a given epoch

    Methods
    -------
    update()
        Appends the current value of the statistic to the time series
    get()
        Returns the time series of statistics
    last_value()
        Returns the last value of the time series of statistics
    minimum()
        Returns the minimum of the time series
    is_minimum()
        Checks if the last value of the time series is strictly less than all the previous values
    """

    def __init__(self, statistic, train=True):
        """Creates an instance of Series

        Args:
            statistic (Statistic): the underlying statistic whose time series we compute during training
            train (bool, optional): whether this is a time series during training (training time) or evaluation (test time). Default: True
        """

        self._statistic = statistic
        self._time_series = []

        self._name = None
        if self._statistic.display_name:
            name = self._statistic.name
            if train: name += '/train'
            else: name += '/test'
            if self._statistic.option: name += '_'+self._statistic.option
            self._name = name

    @property
    def name(self):
        """Gets the name of the time series"""
        return self._name

    def update(self):
        """Appends the current value of the statistic to the time series"""
        self._time_series.append(self._statistic.get())

    def get(self):
        """Returns the time series"""
        return self._time_series

    def last_value(self):
        """Returns the last value of the time series"""
        return self._time_series[-1]

    def minimum(self):
        """Returns the minimum of the time series"""
        return min(self._time_series)

    def is_minimum(self):
        """Checks if the last value of the time series is strictly less than all the previous values

        Returns:
            bool: whether or not the last value of the time series is the strict minimum of the series
        """

        epoch = len(self._time_series)
        if epoch <= 1: return True

        last_value = self._time_series[-1]
        min_value = min(self._time_series[:-1])
        return last_value < min_value

