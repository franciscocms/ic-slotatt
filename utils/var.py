class Variable():
  def __init__(self, name, value, prior_distribution, proposal_distribution, address, instance):
    self._name = name
    self._value = value
    self._prior_distribution = prior_distribution
    self._proposal_distribution = proposal_distribution
    self._address = address
    self._instance = instance
  
  @property
  def name(self):
    return self._name

  @property
  def value(self):
    return self._value
  
  @property
  def prior_distribution(self):
    return self._prior_distribution

  @property
  def proposal_distribution(self):
    return self._proposal_distribution
  
  @property
  def address(self):
    return self._address

  @property
  def instance(self):
    return self._instance