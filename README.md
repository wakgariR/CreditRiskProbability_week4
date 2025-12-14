## Credit Scoring Business Understanding

### Basel II, Risk Measurement, and Model Interpretability

The Basel II Accord places strong emphasis on accurate risk measurement, transparency, and regulatory accountability. Financial institutions must be able to **explain how credit risk is measured, how decisions are made, and why a model produces a specific outcome**. As a result, credit scoring models must be **interpretable, auditable, and well-documented**, allowing regulators, internal auditors, and business stakeholders to understand variable impacts, assumptions, and limitations. Models that cannot be clearly explained may fail regulatory review, regardless of their predictive performance.

### Use of a Proxy Default Variable

Because a direct **default** label is unavailable in the dataset, a **proxy variable** must be constructed (e.g., using delinquency, missed payments, or adverse account behavior). This proxy enables supervised learning but introduces **business risk**, as it may not perfectly represent true default behavior. If the proxy is poorly defined, the model may **misclassify customers**, leading to incorrect credit decisions, unfair customer treatment, increased credit losses, and potential regulatory concerns. Therefore, proxy design must be carefully justified, validated, and clearly communicated to stakeholders.

### Model Choice Trade-offs in a Regulated Environment

There is a fundamental trade-off between **interpretability and predictive power** in credit risk modeling:

- **Simple models** (e.g., Logistic Regression with Weight of Evidence) are highly interpretable, stable, and regulator-friendly. They support transparent decision-making, easier validation, and ongoing monitoring but may sacrifice some predictive accuracy.
- **Complex models** (e.g., Gradient Boosting) often deliver higher predictive performance but behave as **black boxes**, making them difficult to explain, validate, and govern.

In a regulated financial context, **model transparency, governance, and compliance often outweigh marginal performance improvements**, making simpler, well-understood models the preferred choice for production credit scoring systems.
