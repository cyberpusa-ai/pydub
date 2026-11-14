from setuptools import setup

with open("README.markdown", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='pydub-lts',
    version='0.26.0.post3',
    author='James Robert',
    author_email='jiaaro@gmail.com',
    maintainer='Cyber Pusa',
    maintainer_email='packages@cyberpusa.org',
    description='Maintained fork of pydub — drop-in replacement (import pydub), works on Python 3.13+ (no more "No module named \'pyaudioop\'"), fixes int bitrate export, missing-audio and dash-filename crashes.',
    license='MIT',
    keywords='pydub pydub-lts audio sound ffmpeg python3.13 audioop audioop-lts mp3 wav',
    url='https://github.com/cyberpusa-ai/pydub',
    project_urls={
        'Source': 'https://github.com/cyberpusa-ai/pydub',
        'Changelog': 'https://github.com/cyberpusa-ai/pydub/blob/master/CHANGELOG.md',
        'Original project': 'https://github.com/jiaaro/pydub',
    },
    packages=['pydub'],
    python_requires='>=3.7',
    # Python 3.13 removed audioop (PEP 594); audioop-lts is its C build. Without it pydub
    # falls back to the slower pure-Python pydub/pyaudioop.py.
    install_requires=['audioop-lts; python_version >= "3.13"'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Conversion",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Multimedia :: Sound/Audio :: Mixers",
        "Topic :: Software Development :: Libraries",
        'Topic :: Utilities',
    ]
)
