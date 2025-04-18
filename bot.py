import discord
import os
import aiohttp
import json
import logging
from collections import deque

# --- 配置 ---
# 从环境变量获取 Token 和 Key
# 在 Railway 上，你需要在 Variables 设置中添加这些
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# 你可以更改想要使用的 DeepSeek 模型
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat") 
# 触发机器人的方式：'mention' (提及) 或 'prefix' (前缀)
TRIGGER_MODE = os.getenv("TRIGGER_MODE", "mention").lower() 
# 如果 TRIGGER_MODE 是 'prefix', 使用这个前缀
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!ask ") 
# 允许的最大对话历史轮数 (用户 + 机器人)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10")) # 存储最近 5 次用户提问和 5 次机器人回复

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- DeepSeek API 请求函数 ---
async def get_deepseek_response(session, api_key, model, messages):
    """异步调用 DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        # "stream": False, # 如果需要流式输出可以设为 True，但处理会更复杂
        # "max_tokens": 1024, # 可以根据需要限制输出长度
        # "temperature": 0.7, # 控制创造性，值越高越随机
    }

    logger.info(f"Sending request to DeepSeek API with {len(messages)} messages.")
    # logger.debug(f"Request payload: {json.dumps(payload, indent=2)}") # 取消注释以查看详细请求

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
            response_data = await response.json()
            # logger.debug(f"Received response: {json.dumps(response_data, indent=2)}") # 取消注释以查看详细响应

            if response.status == 200:
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0].get("message", {}).get("content")
                    if content:
                        logger.info("Successfully received response from DeepSeek.")
                        return content.strip()
                    else:
                        logger.error("DeepSeek API response missing content.")
                        return "抱歉，DeepSeek API 返回的数据似乎不完整。"
                else:
                    logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                    return f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"
            else:
                error_message = response_data.get("error", {}).get("message", "未知错误")
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}")
                return f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Network connection error: {e}")
        return f"抱歉，无法连接到 DeepSeek API：{e}"
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON response from DeepSeek API.")
        raw_text = await response.text()
        logger.error(f"Raw response text: {raw_text[:500]}...") # 记录部分原始响应
        return "抱歉，无法解析 DeepSeek API 的响应。"
    except Exception as e:
        logger.exception("An unexpected error occurred during DeepSeek API call.")
        return f"抱歉，处理 DeepSeek 请求时发生未知错误: {e}"

# --- Discord 机器人 ---

# 设置 Intents (必须启用 Message Content Intent)
# 在 Discord Developer Portal -> Bot -> Privileged Gateway Intents 中开启
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True # 重要！需要显式启用
intents.guilds = True

client = discord.Client(intents=intents)

# 用于存储每个频道的对话历史 (channel_id -> deque)
conversation_history = {}

@client.event
async def on_ready():
    """机器人启动时执行"""
    logger.info(f'机器人已登录为 {client.user}')
    logger.info(f'使用的模型: {DEEPSEEK_MODEL}')
    logger.info(f'触发模式: {TRIGGER_MODE}')
    if TRIGGER_MODE == 'prefix':
        logger.info(f'命令前缀: "{COMMAND_PREFIX}"')
    logger.info(f'最大历史轮数: {MAX_HISTORY}')
    print("------")
    print(f'Logged in as {client.user.name} ({client.user.id})')
    print(f'Discord.py Version: {discord.__version__}')
    print("Bot is ready!")
    print("------")

@client.event
async def on_message(message):
    """收到消息时执行"""
    # 1. 忽略机器人自己的消息
    if message.author == client.user:
        return

    # 2. 检查触发条件
    triggered = False
    user_prompt = ""

    if TRIGGER_MODE == 'mention':
        # 检查是否提到了机器人
        if client.user.mentioned_in(message):
            triggered = True
            # 移除提及部分，获取实际的用户输入
            # 处理 <@!USER_ID> 和 <@USER_ID> 两种格式
            user_prompt = message.content.replace(f'<@!{client.user.id}>', '', 1).replace(f'<@{client.user.id}>', '', 1).strip()
            if not user_prompt: # 如果用户只@了机器人，没有其他内容
                 await message.channel.send("你好！有什么可以帮你的吗？请 @我 并输入你的问题。", reference=message, mention_author=True)
                 return
    elif TRIGGER_MODE == 'prefix':
        # 检查是否以前缀开头
        if message.content.startswith(COMMAND_PREFIX):
            triggered = True
            user_prompt = message.content[len(COMMAND_PREFIX):].strip()
            if not user_prompt: # 如果用户只输入了前缀
                 await message.channel.send(f"请输入内容。用法：`{COMMAND_PREFIX}你的问题`", reference=message, mention_author=True)
                 return
                 
    # 3. 如果没有触发，直接返回
    if not triggered:
        return

    # 4. 获取或创建频道历史
    channel_id = message.channel.id
    if channel_id not in conversation_history:
        # 使用 deque 来自动管理历史长度
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)

    # 5. 准备发送给 DeepSeek 的消息列表 (包含历史)
    history_deque = conversation_history[channel_id]
    api_messages = list(history_deque) # 从 deque 转换成 list
    api_messages.append({"role": "user", "content": user_prompt})

    # 6. 显示 "正在输入..." 状态并调用 API
    async with message.channel.typing():
        # 使用 aiohttp session 来管理连接
        async with aiohttp.ClientSession() as session:
            logger.info(f"User prompt in channel {channel_id}: {user_prompt}")
            response_content = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

    # 7. 处理和发送回复
    if response_content:
        # 将当前的用户提问和机器人回复添加到历史记录
        history_deque.append({"role": "user", "content": user_prompt})
        history_deque.append({"role": "assistant", "content": response_content})
        
        # 处理 Discord 消息长度限制 (2000 字符)
        if len(response_content) <= 2000:
            await message.channel.send(response_content, reference=message, mention_author=True)
        else:
            # 分割长消息
            logger.warning(f"Response is too long ({len(response_content)} chars), splitting.")
            parts = []
            while len(response_content) > 0:
                # 找到最后一个换行符或空格进行分割，避免截断单词或代码块
                cut_off = 1990 # 留一点余地
                if len(response_content) <= cut_off:
                    parts.append(response_content)
                    response_content = ""
                else:
                    split_index = -1
                    # 优先找换行符
                    try:
                        split_index = response_content[:cut_off].rindex('\n')
                    except ValueError:
                        # 找不到换行符，找空格
                        try:
                            split_index = response_content[:cut_off].rindex(' ')
                        except ValueError:
                            # 连空格都找不到，硬切
                            split_index = cut_off

                    parts.append(response_content[:split_index])
                    response_content = response_content[split_index:].lstrip() # lstrip移除分割后可能产生的行首空格

            for i, part in enumerate(parts):
                # 第一个分段引用原始消息，后续分段不引用
                ref = message if i == 0 else None
                mention = True if i == 0 else False
                await message.channel.send(part, reference=ref, mention_author=mention)
                # 可以加个短暂延迟避免速率限制，但通常不需要
                # await asyncio.sleep(0.5)
    else:
        # 如果 get_deepseek_response 返回空或 None (虽然我们已经处理了错误情况)
        await message.channel.send("抱歉，无法从 DeepSeek 获取有效的回复。", reference=message, mention_author=True)

# --- 运行 Bot ---
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN is None:
        logger.critical("错误：未找到 Discord Bot Token (环境变量 DISCORD_BOT_TOKEN)")
        exit("请设置 DISCORD_BOT_TOKEN 环境变量")
    if DEEPSEEK_API_KEY is None:
        logger.critical("错误：未找到 DeepSeek API Key (环境变量 DEEPSEEK_API_KEY)")
        exit("请设置 DEEPSEEK_API_KEY 环境变量")

    try:
        logger.info("尝试启动 Discord 机器人...")
        client.run(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.critical("Discord Bot Token 无效，登录失败。请检查你的 Token。")
    except discord.PrivilegedIntentsRequired:
        logger.critical("Message Content Intent 未启用！请前往 Discord Developer Portal -> Bot -> Privileged Gateway Intents 开启 Message Content Intent。")
    except Exception as e:
        logger.exception("启动机器人时发生未捕获的错误。")