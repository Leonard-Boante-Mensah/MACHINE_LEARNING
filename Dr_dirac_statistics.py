import numpy as np

# P(A|B) = P(knows the material | answers correctly)


# P(A) = P(knows the material) = 0.6
# P(B) = P(answers correctly) = 0.59

# P(B|A) = P(answers correctly | knows the material) = 0.85

# P(answers correctly | knows the material) * P(knows the material) OR (+) P(answers correctly | does not know the material) * P(does not know the material) = 0.59

# print 0.85 * 0.6 + 0.2 * 0.4

# P(A|B) = P(B|A) * P(A) / P(B)

p_knows_the_material_give_answers_correctly = 0.85 * 0.6 / 0.59

print(p_knows_the_material_give_answers_correctly)

"There is an 86% chance the student really knows the material"