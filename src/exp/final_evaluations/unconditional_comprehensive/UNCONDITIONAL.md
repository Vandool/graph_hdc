Unconditional Generation (Distribution Learning)

Your goal here is to demonstrate that your model can learn the underlying distribution of a training set. You should run this benchmark on both ZINC250K and QM9.

For this task, you must generate a set of 10,000 valid, unique molecules and evaluate them using the following five metrics, as specified in the paper:

    1. Validity:

        Instruction: Generate 10,000 molecules.

        Metric: The ratio of molecules that can be successfully parsed by RDKit.

2. Uniqueness:

   Instruction: Sample from the model until 10,000 valid molecules are obtained.

Metric: The number of unique canonical SMILES strings divided by 10,000.

3. Novelty:

   Instruction: Sample from the model until 10,000 unique, valid molecules are obtained .

Metric: The ratio of these molecules that are not present in your training set.

4. KL Divergence:

   Instruction: Compare the probability distributions of your generated set against a reference set (i.e., your training data).

Metric: Calculate the KL divergence for the 9 physicochemical descriptors listed in the paper (BertzCT, MolLogP, MolWt, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds, NumAliphaticRings, NumAromaticRings) and the distribution of maximum nearest neighbor similarities . The final score S is the average of exp(-D_KL) for each descriptor.

5. Fréchet ChemNet Distance (FCD):

   Instruction: Generate 10,000 valid molecules and compare them to a random 10,000-molecule subset of your training data.

Metric: Use the FCD package to calculate the FCD score. The final benchmark score S is calculated as S = exp(-0.2 * FCD).