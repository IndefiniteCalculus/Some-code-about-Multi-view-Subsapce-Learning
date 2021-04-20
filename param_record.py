# mnist-usps tr 70 te 30
# algorithm="LPDP", t = 1, sigma = 200, lambda1=0.06, lambda2=0.02, lambda3=5e-4, lambda4=5e-4 #  7-10
# algorithm="LPP", sigma = 200, lambda1=0.06, lambda2=0.02, lambda3=5e-3 82.33 # 19
# algorithm="LPP", t = 1, sigma = 200, lambda1=0.00, lambda2=0.00, lambda3=0, lambda4=0 33.83 #
#
# mnist-usps tr 50 te 25
# algorithm="LPDP", t = 1, sigma = 200, lambda1=0.06, lambda2=0.2, lambda3=5e-5, lambda4=5e-5 # 6s 81.8%
# algorithm="LPP", sigma = 200, lambda1=0.6, lambda2=0.02, lambda3=5e-2 # /k=1 5s 82.0% 6590 73% / k=5 21s 81.2% 6462 73% / k=7 21s 81.4% 6494 73%
# algorithm="LPDP", t = 1, sigma = 200, lambda1=0.6, lambda2=0.2, lambda3=0, lambda4=5e-4 # / k=1 7-8s 82.2% 6622 74% / k=3 7s 82.2% 6622 74.8% / k=5 7s 82.6% 6686 75% / k=7 7s 83.8% 6882 76.4%
# algorithm="MMMC", t = 1, sigma = 200, lambda1=0.06, lambda2=0.2, lambda3=0, lambda4=5e-4 # / k=1 8s 82% 6590 72% / k=3 7s 82.6% 6687 73% / k=5 8s 83.8% 6883 75% / k=7 84.4% 6981 75%
#
#

# pie                                                                           # time acc nmi
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-5 # 19s 72.3% 89.6%
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-4 # 19s 72.3% 89.6%
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.002, lambda3=5e-4 # /k=1 70s 79%  91% /k=3 77%  90% /k=5 70s 76%  89% /k=7 69s 73.9%  88%
# algorithm="LPP", t = 1 ,sigma = 2000, lambda1=0.6, lambda2=0, lambda3=2e-0, lambda4=0 # 120s 71% 88%

# algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-5, lambda4=5e-5 # 20s 72.44% 89.6%
# algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-5 # 20s 72.44% 89.6%
# algorithm="LPDP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.002, lambda3=5e-6, lambda4=5e-5 # 87s 79% 91%
# algorithm="LPDP", t = 2, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=5e-3, lambda4=5e-3 # /k=1 85s 79.4% 92% /k=3 78.42% /k=5 76.57% /k=7 75%
#

# algorithm="MMC", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 60s 77.99% 91.4%
# algorithm="MMMC", t = 0.1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 41s 75.33% 90.5%
# algorithm="MMMC", t = 0.01, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 40s 74.67% 90.4%
# algorithm="MMMC", t = 1.5, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # 75s 78.75% 91.5%
# algorithm="MMMC", t = 2, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0, lambda4=5e-3 # /k=1 87s 79.29% 91.8% /k=3 78.09% /k=5 76.35% /k=7 74%
#
# algorithm="LPP", t = 1 ,sigma = 2000, lambda1=0.6, lambda2=0.002, lambda3=0, lambda4=0 # /k=1 75s 79% 91%/k=3 77.88 /k=5 76.03% /k=7 73%
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.6, lambda2=0.02, lambda3=0 # 20s 72.34% 89.6%
# algorithm="LPP", t = 1, sigma = 2000, lambda1=0.06, lambda2=0.02, lambda3=0 # 22s 49.56% 80.93%
#