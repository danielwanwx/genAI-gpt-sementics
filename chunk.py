import re
import utils
import tiktoken

MAX_TOKEN = 512
MAX_HEADING_LEVEL = 6

# Constants
CHUNK_SIZE = 1024  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 256  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
# EMBEDDINGS_BATCH_SIZE = 128  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 4096  # The maximum number of chunks to generate from a text


def get_text_chunks(text: str, chunk_token_size=CHUNK_SIZE):
    """
    Split a text into chunks of ~CHUNK_SIZE tokens, based on punctuation and newline boundaries.

    Args:
        text: The text to split into chunks.
        chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.

    Returns:
        A list of text chunks, each of which is a string of ~CHUNK_SIZE tokens.
    """
    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return []

    # Tokenize the text
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = tokenizer.encode(text)

    # Initialize an empty list of chunks
    chunks = []

    # Use the provided chunk token size or the default one
    chunk_size = chunk_token_size or CHUNK_SIZE

    # Initialize a counter for the number of chunks
    num_chunks = 0

    # tokens of text less than CHUNK_SIZE, then do not divide
    if len(tokens) < CHUNK_SIZE:
        chunks.append(text.replace("\n", " ").strip())
        return chunks

    # Loop until all tokens are consumed
    while tokens and num_chunks < MAX_NUM_CHUNKS:
        # Take the first chunk_size tokens as a chunk
        chunk = tokens[:chunk_size]

        # Decode the chunk into text
        chunk_text = tokenizer.decode(chunk)

        # Skip the chunk if it is empty or whitespace
        if not chunk_text or chunk_text.isspace():
            # Remove the tokens corresponding to the chunk text from the remaining tokens
            tokens = tokens[len(chunk) :]
            # Continue to the next iteration of the loop
            continue

        # Find the last period or punctuation mark in the chunk
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )

        # If there is a punctuation mark, and the last punctuation index is before MIN_CHUNK_SIZE_CHARS
        if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
            # Truncate the chunk text at the punctuation mark
            chunk_text = chunk_text[: last_punctuation + 1]

        # Remove any newline characters and strip any leading or trailing whitespace
        chunk_text_to_append = chunk_text.replace("\n", " ").strip()

        if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            # Append the chunk text to the list of chunks
            chunks.append(chunk_text_to_append)

        # Remove the tokens corresponding to the chunk text from the remaining tokens
        tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())) :]

        # Increment the number of chunks
        num_chunks += 1

    # Handle the remaining tokens
    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(remaining_text)

    return chunks

def get_chunks_by_heading(
    text: str,
    max_level: int
):
    """
    Extract the sections of a Wikipedia page, discarding the references and other low information sections
    """
    # find all headings and the coresponding contents
#     regex_pattern = r'^#{max_level}\s.*$'
    regex_pattern = fr'^#{{{max_level}}}\s.*$'
    headings = re.findall(regex_pattern, text, re.MULTILINE)
    for heading in headings:
        text = text.replace(heading, "==+ !! ==+"+heading)


    contents = text.split("==+ !! ==+")

    return [x for x in contents if len(x.strip()) > 0]

def merge_short_paragraphs(contents, max_token=MAX_TOKEN, direction="bottom-up"):
    results = []
    i = 0
    while i < len(contents):
        if utils.num_tokens_from_string(contents[i]) < max_token:
            j = i + 1
            while j < len(contents) and utils.num_tokens_from_string(contents[i] + contents[j]) <= max_token:
                contents[i] += contents[j]
                j += 1
            results.append(contents[i])
            i = j
        else:
            results.append(contents[i])
            i += 1
    return results

def recursive_split_by_heading(markdown, max_level=1):
    # 如果整篇文档的token小于max_token，则直接返回
    token_count = utils.num_tokens_from_string(markdown)
    if token_count < MAX_TOKEN:
        yield markdown
        return
    
    # 如果 max_level 大于6，直接返回整个 markdown
    if max_level > MAX_HEADING_LEVEL:
        if token_count > CHUNK_SIZE:
            yield from get_text_chunks(markdown)
        else:
            yield markdown
        return

    # 调用 split_by_heading 函数进行分块
    blocks = get_chunks_by_heading(markdown, max_level)
    
    # 调用 merge_short_paragraphs 进行合并
    merged_blocks = merge_short_paragraphs(blocks)

    # 检查分块结果的长度
    for i, block in enumerate(merged_blocks):
        token_count = utils.num_tokens_from_string(block)
#         print('level:', max_level)
#         print('block:', block)
#         print('token_count:', token_count)
#         print('-'*50)
        if token_count > MAX_TOKEN:
            # 如果长度大于50个字符，递归调用 recursive_split_by_heading 函数
            yield from recursive_split_by_heading(block, max_level + 1)
        else:
            yield block

#     # 将分块结果扁平化为一个列表并返回
#     result = []
#     for block in blocks:
#         if isinstance(block, list):
#             result.extend(block)
#         else:
#             result.append(block)

#     return result