from setuptools import setup, find_packages

setup(
    name='fastMPS',
    version='1.0.0',
    description='Fast P(k,z) emulator using t²-PCA and Neural Networks.',
    author='Victoria Lloyd',
    packages=find_packages(),
    install_requires=[
        # Tightly pinned versions to match your working environment (1.26.4)
        # to ensure compatibility with saved joblib/PCA objects.
        'numpy==1.26.4', 
        
        # Pinning joblib to prevent serialization mismatches with scikit-learn
        'joblib', 
        
        # Tightly pinned version to match your working environment (1.5.1)
        # This is critical for loading saved scikit-learn objects (PCA, Scaler).
        'scikit-learn==1.5.1',
        
        'scipy',
        
        # Tightly pinned version to match your working environment (2.19.0)
        # This is critical for loading the Keras model.
        'tensorflow==2.19.0',
        
        # Looser constraint for Colossus, as it's less likely to break serialization
        'colossus>=1.2',
        
    ],
    # Include the model and metadata files
    package_data={
        # Assumes models and metadata are inside the installed package
        'pk_emulator': ['models/*', 'metadata/*'],
    },
    include_package_data=True,
    zip_safe=False,
)
