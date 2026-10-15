# Co-payment and Cost-Share Schedule

**MERIDIAN HEALTH PARTNERS**  
**OPERATIONS POLICY MANUAL**  
**SECTION 4: CLAIMS, BILLING, AND COST-SHARE SCHEDULE**  
**DOCUMENT ID: MHP-POL-412**  
**EFFECTIVE DATE: JANUARY 1, 2025**  
**SUBJECT: BENEFIT PLAN COST-SHARE SCHEDULE AND FEE CALCULATION RULES**

---

### 4.1 Purpose and Scope
This policy defines the standard cost-share figures, fee calculation algorithms, and co-pay structures for all health benefit plans administered across the 14 Meridian Health Partners outpatient clinic sites. All front-desk staff, billing specialists, and the automated claims adjudication engine (Platform ID: Clamsys-v4) must strictly apply the figures and rules contained herein. No exceptions or local modifications are permitted without the written authorization of the Chief Financial Officer, Marcus Vance.

---

### 4.2 Cost-Share Schedule by Plan
The automated claims system holds and enforces the exact financial parameters detailed in Table 4.1 for each of the four Meridian-administered plans.

#### Table 4.1: Plan Cost-Share Parameters
| Plan Identifier | Annual Deductible | Coinsurance Rate | Out-of-Pocket Maximum | Specialist Co-Pay | Telehealth Co-Pay | Imaging Co-Pay | Lab Work Co-Pay |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MERIDIAN-GOLD** | $500 | 20% | $25,000 | $25 | $10 | $60 | $15 |
| **MERIDIAN-SILVER** | $1,500 | 30% | $15,000 | $45 | $18 | $110 | $25 |
| **MERIDIAN-BRONZE** | $3,000 | 40% | $8,000 | $45 | $18 | $160 | $35 |
| **CIVIC-BASE** | $2,000 | 35% | $10,000 | $45 | $18 | $130 | $30 |

---

### 4.3 Telehealth Co-Pay Derivation Rule
1. Telehealth visits across all plans carry a flat co-pay equal to exactly 40 percent of the respective plan's in-person specialist co-pay.
2. System rounding shall be rounded to the nearest whole dollar. Applying this rule yields the exact telehealth co-pays listed in Table 4.1 ($10 for MERIDIAN-GOLD; $18 for MERIDIAN-SILVER, MERIDIAN-BRONZE, and CIVIC-BASE).

---

### 4.4 Claims Calculator Calculation Rules
The claims system calculator must apply the following computational logic when determining patient financial responsibility for outpatient encounters at any Meridian Health Partners facility:

1. **Consultations, Imaging, and Lab Services:**  
   A standard consultation, imaging service, or laboratory service costs the patient the flat co-pay amount specified in Table 4.1 only. Deductible and coinsurance rules do not apply to these specific service types when billed as standalone line items.

2. **Procedures:**  
   When a service is classified as a procedure (CPT code ranges 10000–69999 and designated hospital outpatient surgical codes), the patient owes:
   * The unmet portion of the annual deductible, **plus**
   * The plan's designated coinsurance rate applied to the remainder of the list price (List Price minus Unmet Deductible).
   * **Absolute Ceiling:** The total patient responsibility for any single encounter or cumulative annual claims shall never exceed the plan's designated out-of-pocket maximum.

3. **Deductible Status Verification:**  
   If the claims system records a deductible-met amount that is equal to or greater than the plan's stated annual deductible, the deductible is deemed fully met, and the unmet deductible component is reduced to $0.00.

---

### 4.5 Worked Example: Partial Deductible Procedure Calculation
The claims calculator executes the following steps for a procedural claim where the patient's deductible is partially met. 

* **Patient Plan:** MERIDIAN-BRONZE  
* **Annual Deductible:** $3,000  
* **Deductible Currently Met:** $1,800  
* **Coinsurance Rate:** 40%  
* **Encounter Details:** Knee Arthroscopy (Service Code: ORTH-ARTH), List Price: $6,200  

#### Calculation Steps:
1. **Determine Unmet Deductible:**  
   $$\text{Total Deductible } (\$3,000) - \text{Deductible Met } (\$1,800) = \$1,200 \text{ unmet deductible}$$
2. **Calculate Coinsurance Base:**  
   $$\text{List Price } (\$6,200) - \text{Unmet Deductible } (\$1,200) = \$5,000 \text{ remainder}$$
3. **Apply Coinsurance Rate:**  
   $$40\% \text{ of } \$5,000 = \$2,000 \text{ coinsurance amount}$$
4. **Sum Total Patient Share:**  
   $$\text{Unmet Deductible } (\$1,200) + \text{Coinsurance } (\$2,000) = \mathbf{\$3,200 \text{ total patient owing}}$$

The claims system shall generate a patient invoice for exactly $3,200, provided this amount does not breach the $8,000 out-of-pocket maximum for the MERIDIAN-BRONZE plan.

---

### 4.6 Administrative Compliance and Audit
Any manual override of the automated co-pay schedule or calculator rules requires dual sign-off from the Billing Operations Manager, Sarah Jenkins, and the Revenue Cycle Director, David Thorne. Monthly audit logs of all override transactions are submitted directly to the Office of Compliance.
