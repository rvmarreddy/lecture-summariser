# W1 L2: Data representation + Ingestion

1 October 2025 

[cm50266_week1_L2_data_ingestion25.pdf](W1%20L2%20Data%20representation%20+%20Ingestion/cm50266_week1_L2_data_ingestion25.pdf)

## Data Types

<aside>
💡

**Quantitative:** data that is countable and measurable

- Unique mapping of property to numerical value
    - there is no ambiguity in the results, not based on opinion or confusion
- Consistently measurable within some precision

<aside>
📎

e.g. how long your coursework takes.

e.g. percentage chance of rain: 30%

</aside>

</aside>

<aside>
💡

**Qualitative:** data that records a quality

- this could be a characteristic and data collected is often subjective

<aside>
📎

e.g. how happy are you today?

- the recorded data could be a subjective scale
</aside>

</aside>

---

## Representation

The datatypes need to be stored in a computer, and it works in bits.

<aside>
💡 **A byte** is the smallest chunk of memory, made up of 8 bits, that computers can directly access using an address.

- If your computer has 1GB of RAM, it has 1 billion uniqes addresses each mapping to one byte
</aside>

### Integers

- in multiples of 8 bits (byte addressable)
    - Larger quantities are typically size aligned
        
        <aside>
        📎 Take a 4-byte integer fits neatly in a memory shelf that starts at an address that is a multiple of its size. e.g. using consecutive addresses: 0x1000 - 0x1003.
        
        - unaligned addresses 0x1001 - 0x1004. The CPU fetches two memory shelves 0x1000-0x1003 and 0x1004-0x1007, then combines them.
            - CPU is fast temporary storage for running programs. Works best with aligned data.
        - Files can store data tightly packed.
            - space efficiency is prioritised over alignment
        </aside>
        
- The internal datapath of modern processors typically 32-128 bit
    - A 32-bit data path can fetch and compute a 32-bit integer in a single step
- They usually support the lower widths
    - CPU can efficiently handle smaller integers within the same data path.
    - These may be multiple at the same time.
        - e.g. 64-bit processor could potentially: process two 32-bit integers or four 16-bit integers.

- Unsigned represents positive numbers only
    - 16 bits gives 0 - 65535 range = $2^{16}-1$

- negative numbers are inverted binary +1.
    - e.g. 0101 = 5 —> invert 1010 —> +1 —> 1011 = -5

![Screenshot 2025-10-03 at 16.33.28.png](W1%20L2%20Data%20representation%20+%20Ingestion/Screenshot_2025-10-03_at_16.33.28.png)

- Looking at a 3 bit integer, the first digit determines whether it is positive or negative.
    - Done using this method since addition and subtraction cancel 0101 + 1011 = 0000 (ignoring overflow)
    - CPU knows the width of the number because the instruction specifies it.

### Real Numbers

- Computers aren’t able to store real numbers.
    - The closest they come is floating point.
    - To save on architecture
- Most follow IEEE754 standard is used
    - Watch out for compatible vs compliant vs fully compliant

<aside>
💡

**Compatible:** Systems that implement the basic formats but might not include all required operations or might handle some edge cases differently.

**Compliant:** Systems that implement both the required formats and operations, but might not include all recommended features.

**Fully compliant:** Systems that implement all required and recommended aspects of the standard.

</aside>

### **IEEE 754**

- floating point representation:

![Screenshot 2025-10-01 at 17.48.27.png](W1%20L2%20Data%20representation%20+%20Ingestion/Screenshot_2025-10-01_at_17.48.27.png)

- numbers supported depends on number of bits allocated to it
    - sign (1 bit); 0 = positive, 1 = negative
    - Mantissa determines the precision.
    - Exponent stored with a bias so that both positive and negative exponents can be represented.
    - Sign is always 1 bit. Sign magnitude values.
    - Bias is a fixed value, depending on bit-width of exponent.

<aside>
📎

**IEEE 754: converting -6.75**

- Sign bit: 1
- Integer part: 6 = 110
- fractional part: calculate by multiplying by 2 repeatedly and noting the integer value.
- 0.75 = 0.11
- 6.75 = 110.11 in binary

**Normalised**

110.11 → 1.1011 x 2^2

- Mantissa = bits after the decimal: 10110000000000000000000 (padded to 23 bits)
- Exponent = 2
- For 32-bit floats, the bias is 127, $2^8 -1$ (8 bits for exponent)
    - stored exponent = actual exponent + bias
    - if the actual exponent is 2 → store 2 + 127 = 129
    - in binary that is 10000001
- 1 10000001 10110000000000000000000
</aside>

**Special cases**

- **Zero:** Exp = 0, Mant = 0
- **De-normalised numbers:** Exp = 0, Matt != 0
- **+/- Infinity:** Exp = all 1, Mant = 0
- **NaN**, **Not a number:** Exp = all 1, Mant != 0

- 32 and 64 bit are most common
    - 80 bit may be used internally. GPUs/ML specific often support 16 bit
- Multiple rounding modes
    - the default, nearest is most commonly used.
- The order of operations will matter
    - log values used to manage multiples hitting zero so quickly

### Strings

- To understand which method is used to represent the strings. You should look out for:
    1. Are these stored as length + data or data with end of string marker.
    2. How are the characters represented
- most common representations: ASCII and UTF-8

**ASCII**

- 128 characters (7 bits)
    - 7 bits chosen since it was just enough for the characters required
    - later now we have the extended ASCII using 8-bit, 256 characters
- 96 printable
- This 8th bit was not standardised, so different systems interpreted them differently

**UTF-8**

- Uses between 1 and 4 bytes per character
    - IANA (Internet assigned number authority): is responsible for keeping the internet’s core standards and uniques identifiers consistent
- Writing notation:
    - If the MSB (most significant bit) of first byte is 0, remainder map to ASCII: values between (00000000-01111111)
- Prefix bytes:
    - 2-byte sequence: 110xxxxx (first byte) 10xxxxxx (second byte)
    - 4-byte sequence: 11110xxx (first byte) followed by 3 continuation bytes 10xxxxxx

## In memory and files

- briefly overview

<aside>
💡

**Endianness = byte order**

A 32-bit number takes 4 bytes, and endianness decides whether the lowest or highest 8 bits get stored at the lowest memory address. 

- Little Endian (most common): least significant byte is stored at the lowest address.
- Big Endian is the opposite
    - understand the the most significant byte contains the highest-order bits
    - e.g. 0x12345678, MSB = 0x12 (bits 31-24)
    - lowest address would be 0x1000
- Most CPUs/processors have a fixed endianness but some can switch depending on configuration
- Importance
    - Reading/writing binary file formats that specify a fixed byte order
    - Sending/receiving data over networks
    - Debugging raw memory or transferring data between systems of different endianness.
</aside>

## Data Ingestion

### Sources of Data

**Primary Data**

- Data collected by the researcher
- Often expensive, time consuming
- But specific to the research

**Secondary Data**

- Collected by someone else
- May be outdated
- Saves time/effort
- May not be ideal for research

**Manually**

- Labour intensive
- Prone to human error

**Automatic Collection**

- Risk of being overwhelmed by the data
    - example of the CERN hydron collider generating 1 petabyte per second
    - This is filtered to one petabyte per day but the continuous stream of data requires extreme infrastructure to manage this
- Accurate within measurable tolerances
- Potential for sensor failure or systematic error

**Observed**

- Data collected manually or automatically is observed data

**Synthesis**

- We can create new data by synthesis
    - Manipulate existing data to create new data
    - e.g. flipping or rotating images to create more input
- If we have an existing model it can be used to create new data items consistently with the model
    - Care is required when using this for training
    - Sythesis may be the main objective for example generated higher-resolution images by hallucinating details

### Format of Data

**Physical Data**

- Data on paper forms
- Might be the only practical way to collect due to lack of digital infrastructure
- Often transcribed/scanned into electronic forms
    - ML-based OCR (Optical character recognition
        - uses ML models ro recognis characters and words from scanned images

**Physical data format**

<aside>
📎

**Designing forms**

- Consider the objectives
    - Easy to complete
        - efficiency
        - minimise errors
    - comprehensive
- easy to complete vs comprehensive often mutually exclusive

**Paper forms**

- Labels need to be consistent, descriptive but concise
- clear input mechanism
    - Numbers, strings, ticks, crosses etc.
- Minimum effort to complete
    - no look-ups/translation
    - tick marks instead of distinct entries
    - helps avoid fatigue and reduces error
- Results can be post-processed
    - turn text into integer identifiers. e.g. male, female, other → 0, 1, 2
    - Compute useful values from recorded values e.g. from DoB → age
- How will the form be digitised?
    - humans good at text, numbers
    - computers good at shaded boxes
</aside>

**Electronic Data**

- automatically gathered
- manually input
- or transcribed
    - data electronically means it can be processed by a computer

## Data Wrangling

<aside>
💡

- **Data Wrangling:** process of cleaning, organising and transforming raw data into a format that is ready for analysis.
- This could be through a mixture of manual and automatic processes
    - removing data duplicates or standardising formats
    - correcting typos or filling missing values based on domain knowledge
</aside>

### Data Enrichment

- Generate additional data from existing data
    - combining multiple sources of data
    - Find relationships between data