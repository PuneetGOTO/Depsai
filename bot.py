import discord
from discord.ext import commands
from discord import app_commands # 用于斜杠命令
from discord.ui import View, Button, button # 用于按钮和视图
import os
import aiohttp
import json
import logging
from collections import deque
import asyncio
import re # 用于清理用户名

# --- 提前设置日志记录 ---
# 配置日志记录器，设定级别为 INFO，并指定格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
# 获取当前模块的 logger 实例
logger = logging.getLogger(__name__)

# --- 配置 ---
# 从环境变量获取敏感信息和配置
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions" # DeepSeek API 端点
# 设置默认模型，优先从环境变量读取，否则使用 deepseek-reasoner
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
logger.info(f"Initializing with DeepSeek Model: {DEEPSEEK_MODEL}") # 记录使用的模型
# 设置最大历史记录轮数 (用户+机器人)，优先从环境变量读取，默认 10
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
# 设置分割长消息时的发送延迟（秒），优先从环境变量读取，默认 0.3
SPLIT_MESSAGE_DELAY = float(os.getenv("SPLIT_MESSAGE_DELAY", "0.3"))
# 获取管理员角色 ID 列表（逗号分隔），用于特定权限
admin_ids_str = os.getenv("ADMIN_ROLE_IDS", "")
ADMIN_ROLE_IDS = [int(role_id) for role_id in admin_ids_str.split(",") if role_id.strip().isdigit()]
# 设置私密聊天频道名称的前缀
PRIVATE_CHANNEL_PREFIX = "deepseek-"

# --- DeepSeek API 请求函数 ---
async def get_deepseek_response(session, api_key, model, messages):
    """异步调用 DeepSeek API，处理 reasoning_content 和 content"""
    headers = {
        "Authorization": f"Bearer {api_key}", # API Key 认证
        "Content-Type": "application/json" # 请求体格式
    }
    # 构建请求体
    payload = {
        "model": model, # 使用指定的模型
        "messages": messages, # 包含历史记录的消息列表
        # 注意：根据文档，reasoner 模型不支持 temperature, top_p 等参数
    }
    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")

    try:
        # 发送 POST 请求，设置超时时间 (例如 5 分钟)
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=300) as response:
            # 尝试读取原始响应文本，以防 JSON 解析失败时也能诊断
            raw_response_text = await response.text()
            try:
                # 解析 JSON 响应
                response_data = json.loads(raw_response_text)
            except json.JSONDecodeError:
                 # JSON 解析失败的处理
                 logger.error(f"Failed to decode JSON response from DeepSeek API. Status: {response.status}. Raw text: {raw_response_text[:500]}...")
                 return None, f"抱歉，无法解析 DeepSeek API 的响应 (状态码 {response.status})。"

            # 处理 HTTP 状态码
            if response.status == 200: # 请求成功
                 if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message_data = response_data["choices"][0].get("message", {})
                    # 获取思维链和最终回答
                    reasoning_content = message_data.get("reasoning_content")
                    final_content = message_data.get("content")
                    usage = response_data.get("usage") # 获取 token 使用量

                    # 组合用于 Discord 显示的完整响应
                    full_response_for_discord = ""
                    if reasoning_content: # 如果有思维链，格式化添加
                        full_response_for_discord += f"🤔 **思考过程:**\n```\n{reasoning_content.strip()}\n```\n\n"
                    if final_content: # 如果有最终回答，格式化添加
                        full_response_for_discord += f"💬 **最终回答:**\n{final_content.strip()}"
                    # 处理只有思维链的情况
                    if not final_content and reasoning_content:
                         full_response_for_discord = reasoning_content.strip()
                    # 处理两者皆无的情况
                    if not full_response_for_discord:
                         logger.error("DeepSeek API response missing both reasoning_content and content.")
                         return None, "抱歉，DeepSeek API 返回的数据似乎不完整。"

                    logger.info(f"Successfully received response from DeepSeek. Usage: {usage}")
                    # 返回两个值：用于显示的完整字符串，和仅用于历史的最终内容
                    return full_response_for_discord.strip(), final_content.strip() if final_content else None
                 else: # 响应成功但结构不符合预期 (缺少 choices)
                     logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                     return None, f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"
            else: # 处理非 200 的错误状态码
                error_message = response_data.get("error", {}).get("message", f"未知错误 (状态码 {response.status})")
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}. Response: {raw_response_text[:500]}")
                # 对 400 错误添加额外提示
                if response.status == 400:
                    logger.warning("A 400 error might be caused by including 'reasoning_content' in the input messages.")
                    error_message += "\n(提示: 错误 400 通常因为请求格式错误)"
                # 返回 None 和错误消息
                return None, f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"
    # 处理网络连接、超时等 aiohttp 异常
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Network connection error: {e}")
        return None, f"抱歉，无法连接到 DeepSeek API：{e}"
    except asyncio.TimeoutError:
        logger.error("Request to DeepSeek API timed out.")
        return None, "抱歉，连接 DeepSeek API 超时。"
    # 捕获其他所有未预料的异常
    except Exception as e:
        logger.exception("An unexpected error occurred during DeepSeek API call.")
        return None, f"抱歉，处理 DeepSeek 请求时发生未知错误: {e}"

# --- Discord 机器人设置 ---
# 定义机器人所需的 Intents (权限意图)
intents = discord.Intents.default()
intents.messages = True       # 需要接收消息事件
intents.message_content = True # 需要读取消息内容 (必需！)
intents.guilds = True         # 需要访问服务器信息（如创建频道）
# 创建 Bot 实例 (使用 commands.Bot 以支持斜杠命令)
# command_prefix 在此应用中不重要，因为我们主要用斜杠命令和按钮
bot = commands.Bot(command_prefix="!", intents=intents)

# 对话历史记录 (字典: key=频道ID, value=deque对象存储消息)
conversation_history = {}

# --- 创建按钮视图 ---
# 定义一个包含“创建私密聊天”按钮的视图类
class CreateChatView(View):
    # 注意：当前按钮是非持久化的，机器人重启后会失效
    # 要实现持久化，需要设置 timeout=None 并 在 setup_hook 中处理 bot.add_view()
    @button(label="创建私密聊天", style=discord.ButtonStyle.primary, emoji="💬", custom_id="create_deepseek_chat_button")
    async def create_chat_button_callback(self, interaction: discord.Interaction, button_obj: Button):
        """处理“创建私密聊天”按钮点击事件"""
        guild = interaction.guild # 获取服务器对象
        user = interaction.user   # 获取点击按钮的用户对象
        # 在回调中重新获取最新的 bot member 对象，确保权限信息准确
        bot_member = guild.get_member(bot.user.id) if guild else None

        # 检查是否在服务器内且能获取到机器人成员信息
        if not guild or not bot_member:
            await interaction.response.send_message("无法获取服务器或机器人信息。", ephemeral=True)
            return

        # 检查机器人是否有创建频道的权限
        if not bot_member.guild_permissions.manage_channels:
            await interaction.response.send_message("机器人缺少“管理频道”权限，无法创建聊天频道。", ephemeral=True)
            logger.warning(f"Button Click: Missing 'Manage Channels' permission in guild {guild.id}")
            return

        # 清理用户名以创建合法的频道名称
        clean_user_name = re.sub(r'[^\w-]', '', user.display_name.replace(' ', '-')).lower()
        # 防止用户名处理后为空
        if not clean_user_name: clean_user_name = "user"
        # 生成频道名称，包含前缀、清理后的用户名和部分用户ID以减少重名冲突
        channel_name = f"{PRIVATE_CHANNEL_PREFIX}{clean_user_name}-{user.id % 10000}"

        # 设置频道的权限覆盖规则
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=False), # @everyone 角色不可见
            user: discord.PermissionOverwrite(read_messages=True, send_messages=True, view_channel=True), # 点击用户可见可写
            bot_member: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, manage_channels=True) # 机器人需要读写和管理权限
        }
        # 为配置的管理员角色添加权限
        for role_id in ADMIN_ROLE_IDS:
            role = guild.get_role(role_id)
            if role:
                overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, view_channel=True)
            else:
                 logger.warning(f"Admin role ID {role_id} not found in guild {guild.id}")

        try:
            # 先响应交互，告知用户正在处理，防止交互超时 (ephemeral=True 仅点击者可见)
            await interaction.response.send_message(f"正在为你创建私密聊天频道 **{channel_name}** ...", ephemeral=True)

            # 尝试找到名为 "DeepSeek Chats" 的分类
            category_name = "DeepSeek Chats"
            category = discord.utils.find(lambda c: c.name.lower() == category_name.lower(), guild.categories)
            target_category = None # 目标分类，默认为 None (即放在服务器顶层)
            # 如果找到分类，且是分类频道类型，并且机器人有在此分类下管理频道的权限
            if category and isinstance(category, discord.CategoryChannel) and category.permissions_for(bot_member).manage_channels:
                 target_category = category # 将目标分类设为找到的分类
                 logger.info(f"Found suitable category '{category.name}' for new channel.")
            else:
                 # 如果找不到分类或无权限，则记录警告，频道将创建在默认位置
                 if category: logger.warning(f"Found category '{category_name}' but bot lacks permissions or it's not a CategoryChannel. Creating channel in default location.")
                 else: logger.info(f"Category '{category_name}' not found, creating channel in default location.")
                 # 可选：如果需要自动创建分类，可以在这里添加 guild.create_category 代码

            # 创建新的私密文本频道
            new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites, category=target_category)
            logger.info(f"Successfully created private channel {new_channel.name} ({new_channel.id}) for user {user.name} ({user.id})")

            # 在新创建的频道中发送欢迎和引导消息
            welcome_message = (
                f"你好 {user.mention}！\n"
                f"欢迎来到你的 DeepSeek 私密聊天频道 (使用模型: **{DEEPSEEK_MODEL}**)。\n"
                f"直接在此输入你的问题即可开始对话。\n"
                f"对话历史最多保留 **{MAX_HISTORY // 2}** 轮问答。\n"
                f"当你完成后，可以在此频道使用 `/close_chat` 命令来关闭它。"
            )
            await new_channel.send(welcome_message)

            # 使用 followup.send 来在初始响应后发送频道链接 (因为 response 只能调用一次)
            await interaction.followup.send(f"你的私密聊天频道已创建：{new_channel.mention}", ephemeral=True)

        # 处理创建频道过程中可能发生的错误
        except discord.Forbidden: # 机器人缺少权限
            logger.error(f"Button Click: Permission error (Forbidden) for user {user.id} creating channel '{channel_name}'.")
            # 尝试用 followup 回复错误信息
            try: await interaction.followup.send("创建频道失败：机器人权限不足。", ephemeral=True)
            except discord.NotFound: pass # 如果交互已超时或失效
        except discord.HTTPException as e: # Discord API 网络或速率限制错误
            logger.error(f"Button Click: HTTP error for user {user.id} creating channel '{channel_name}': {e}")
            try: await interaction.followup.send(f"创建频道时发生网络错误。", ephemeral=True)
            except discord.NotFound: pass
        except Exception as e: # 其他意外错误
            logger.exception(f"Button Click: Unexpected error for user {user.id} creating channel")
            try: await interaction.followup.send(f"创建频道时发生未知错误。", ephemeral=True)
            except discord.NotFound: pass

# --- setup_hook: 机器人启动时运行 ---
@bot.event
async def setup_hook():
    """在机器人登录后、连接到网关前执行的异步设置"""
    logger.info("Running setup_hook...")
    # 打印配置信息到日志
    logger.info(f'--- Bot Configuration ---')
    logger.info(f'Default/Current DeepSeek Model: {DEEPSEEK_MODEL}')
    logger.info(f'Max Conversation History Turn: {MAX_HISTORY}')
    logger.info(f'Split Message Delay: {SPLIT_MESSAGE_DELAY}s')
    logger.info(f'Admin Role IDs: {ADMIN_ROLE_IDS}')
    logger.info(f'Private Channel Prefix: {PRIVATE_CHANNEL_PREFIX}')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info(f'-------------------------')
    # 同步应用程序命令（斜杠命令）到 Discord
    try:
        # bot.tree.sync() 会将所有定义的命令同步到 Discord
        # 全局同步可能需要一些时间生效（最多1小时）
        synced = await bot.tree.sync()
        # 记录已同步的命令列表
        logger.info(f"Synced {len(synced)} slash commands globally: {[c.name for c in synced]}")
    except Exception as e:
        # 记录同步命令时发生的任何错误
        logger.exception(f"Failed to sync slash commands: {e}")

# --- on_ready: 机器人准备就绪事件 ---
@bot.event
async def on_ready():
    """当机器人成功连接并准备好处理事件时调用"""
    logger.info(f'机器人已登录为 {bot.user}')
    # 在控制台打印简单的就绪信息
    print(f"Logged in as {bot.user} ({bot.user.id})")
    print("Bot is ready and functional.")

# --- 斜杠命令：/setup_panel ---
# 定义一个名为 "setup_panel" 的斜杠命令
@bot.tree.command(name="setup_panel", description="发送一个包含'创建聊天'按钮的消息到当前频道")
async def setup_panel(interaction: discord.Interaction, message_content: str = "点击下面的按钮开始与 DeepSeek 的私密聊天："):
    """发送包含创建聊天按钮的消息，根据频道类型检查权限"""
    channel = interaction.channel # 获取命令执行的频道
    user = interaction.user     # 获取执行命令的用户
    # 检查命令是否在服务器内执行
    if not interaction.guild:
        await interaction.response.send_message("此命令只能在服务器频道中使用。", ephemeral=True)
        return

    # 判断是否在机器人创建的私密频道中执行
    is_private_chat_channel = isinstance(channel, discord.TextChannel) and channel.name.startswith(PRIVATE_CHANNEL_PREFIX)
    can_execute = False # 默认不允许执行

    # 权限检查逻辑
    if is_private_chat_channel:
        # 如果在私密频道中，允许执行 (因为只影响该频道)
        can_execute = True
        logger.info(f"User {user} executing /setup_panel in private channel {channel.name}. Allowed.")
    elif isinstance(user, discord.Member) and user.guild_permissions.manage_guild:
        # 如果在其他频道，需要用户拥有“管理服务器”权限
        can_execute = True
        logger.info(f"User {user} executing /setup_panel in public channel {channel.name}. Allowed (has manage_guild).")
    else:
        # 在其他频道且无权限，发送提示并阻止
        logger.warning(f"User {user} trying to execute /setup_panel in public channel {channel.name} without manage_guild permission. Denied.")
        await interaction.response.send_message("你需要在非私密频道中使用此命令时拥有“管理服务器”权限。", ephemeral=True)
        return # 阻止后续代码执行

    # 如果权限检查通过
    if can_execute:
        try:
            # 创建一个新的按钮视图实例
            view = CreateChatView()
            # 在当前频道发送包含按钮的消息
            await channel.send(message_content, view=view)
            # 对原始交互进行响应，告知用户操作成功 (仅用户可见)
            await interaction.response.send_message("创建聊天按钮面板已发送！", ephemeral=True)
            logger.info(f"User {user} successfully deployed the create chat panel in channel {channel.id}")
        # 处理发送消息时可能发生的错误
        except discord.Forbidden: # 机器人缺少发送消息/添加组件的权限
             logger.error(f"Failed to send setup panel in {channel.id}: Missing permissions.")
             try: await interaction.followup.send("发送失败：机器人缺少在此频道发送消息或添加组件的权限。", ephemeral=True)
             except discord.NotFound: pass
        except Exception as e: # 其他未知错误
            logger.exception(f"Failed to send setup panel in {channel.id}")
            try: await interaction.followup.send(f"发送面板时发生错误：{e}", ephemeral=True)
            except discord.NotFound: pass

# --- 斜杠命令：/clear_history ---
# 定义一个名为 "clear_history" 的斜杠命令
@bot.tree.command(name="clear_history", description="清除当前私密聊天频道的对话历史")
async def clear_history(interaction: discord.Interaction):
    """处理 /clear_history 命令，用于清除当前私密频道的历史"""
    channel = interaction.channel
    user = interaction.user
    # 检查命令是否在正确的频道类型中执行
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        await interaction.response.send_message("此命令只能在 DeepSeek 私密聊天频道中使用。", ephemeral=True)
        return

    channel_id = channel.id
    # 检查内存中是否存在该频道的历史记录
    if channel_id in conversation_history:
        try:
            # 清空 deque 对象
            conversation_history[channel_id].clear()
            logger.info(f"User {user.name} ({user.id}) cleared history for channel {channel.name} ({channel_id})")
            # 公开回复确认历史已清除
            await interaction.response.send_message("当前频道的对话历史已清除。", ephemeral=False)
        except Exception as e:
            # 处理清除过程中可能发生的错误
            logger.exception(f"Error clearing history for channel {channel_id}")
            await interaction.response.send_message(f"清除历史时发生错误：{e}", ephemeral=True)
    else:
        # 如果内存中找不到历史记录
        logger.warning(f"User {user.name} ({user.id}) tried to clear history for channel {channel_id}, but no history found.")
        await interaction.response.send_message("未找到当前频道的历史记录（可能从未在此对话或机器人已重启）。", ephemeral=True)

# --- 斜杠命令：/help ---
# 定义一个名为 "help" 的斜杠命令
@bot.tree.command(name="help", description="显示机器人使用帮助")
async def help_command(interaction: discord.Interaction):
    """处理 /help 命令，发送嵌入式帮助信息"""
    # 创建一个嵌入式消息对象
    embed = discord.Embed(
        title="DeepSeek 机器人帮助",
        description=f"你好！我是使用 DeepSeek API ({DEEPSEEK_MODEL}) 的聊天机器人。",
        color=discord.Color.purple() # 设置侧边颜色
    )
    # 添加字段说明用法
    embed.add_field(name="如何开始聊天", value="点击管理员放置在服务器频道中的 **“创建私密聊天”** 按钮，我会为你创建一个专属频道。", inline=False)
    embed.add_field(name="在私密频道中", value=f"• 直接输入你的问题即可与 DeepSeek 对话。\n• 我会记住最近 **{MAX_HISTORY // 2}** 轮的问答。", inline=False)
    embed.add_field(name="可用命令", value="`/help`: 显示此帮助信息。\n`/clear_history`: (仅在私密频道内可用) 清除当前频道的对话历史。\n`/close_chat`: (仅在私密频道内可用) 关闭当前的私密聊天频道。\n`/setup_panel`: (管理员或私密频道内可用) 发送创建按钮面板。", inline=False)
    # 添加页脚信息
    embed.set_footer(text=f"当前模型: {DEEPSEEK_MODEL} | 由 discord.py 和 DeepSeek 驱动")
    # 发送嵌入式消息，设为 ephemeral (仅命令使用者可见)
    await interaction.response.send_message(embed=embed, ephemeral=True)

# --- 斜杠命令：/close_chat (包含 NameError 修正) ---
# 定义一个名为 "close_chat" 的斜杠命令
@bot.tree.command(name="close_chat", description="关闭当前的 DeepSeek 私密聊天频道")
async def close_chat(interaction: discord.Interaction):
    """处理 /close_chat 命令，包含 NameError 修正"""
    channel = interaction.channel
    user = interaction.user
    # 检查是否在正确的频道类型中
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        await interaction.response.send_message("此命令只能在 DeepSeek 私密聊天频道中使用。", ephemeral=True)
        return

    try:
        # 先响应交互
        await interaction.response.send_message(f"请求收到！频道 {channel.mention} 将在几秒后关闭...", ephemeral=True)
        # 在频道内发送公开通知
        await channel.send(f"此聊天频道由 {user.mention} 请求关闭，将在 5 秒后删除。")
        logger.info(f"User {user.name} ({user.id}) initiated closure of channel {channel.name} ({channel.id})")

        # --- 修正部分：先获取 channel_id 再删除历史 ---
        channel_id = channel.id # 正确获取频道 ID
        if channel_id in conversation_history:
            try:
                del conversation_history[channel_id] # 使用正确的变量名删除历史
                logger.info(f"Removed conversation history for channel {channel_id}")
            except KeyError:
                 # 如果尝试删除时 key 已不存在 (可能被其他方式清除了)
                 logger.warning(f"Tried to delete history for channel {channel_id}, but key was already gone.")
        else:
             # 如果内存中原本就没有这个频道的历史记录
            logger.warning(f"No history found in memory for channel {channel_id} during closure.")
        # --- 修正结束 ---

        # 延迟一段时间给用户看到消息
        await asyncio.sleep(5)
        # 删除频道
        await channel.delete(reason=f"Closed by user {user.name} ({user.id}) via /close_chat")
        logger.info(f"Deleted channel {channel.name} ({channel.id})")

    # 处理删除过程中可能发生的错误
    except discord.Forbidden: # 机器人缺少删除频道的权限
        logger.error(f"Permission error (Forbidden) while trying to delete channel {channel.id}.")
        try: await interaction.followup.send("关闭频道失败：机器人缺少“管理频道”权限。", ephemeral=True)
        except discord.NotFound: pass # 交互可能已失效
    except discord.NotFound: # 尝试删除时频道已不存在
        logger.warning(f"Channel {channel.id} not found during deletion, likely already deleted.")
    except discord.HTTPException as e: # Discord API 错误
        logger.error(f"HTTP error while deleting channel {channel.id}: {e}")
    except Exception as e: # 其他意外错误
        logger.exception(f"Unexpected error during /close_chat command for channel {channel.id}")
        # 尝试回复错误，但此时频道可能已删除
        try: await interaction.followup.send(f"关闭频道时发生意外错误: {e}", ephemeral=True)
        except Exception as fu_e: logger.error(f"Could not send followup error for /close_chat: {fu_e}")

# --- on_message 事件处理 (核心对话逻辑) ---
@bot.event
async def on_message(message: discord.Message):
    """处理在私密频道中接收到的消息"""
    # 1. 忽略机器人自己、其他机器人或 Webhook 发送的消息
    if message.author == bot.user or message.author.bot:
        return
    # 2. 只处理特定前缀的文本频道（即我们创建的私密频道）
    if not isinstance(message.channel, discord.TextChannel) or not message.channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        return
    # 3. 忽略空消息或看起来像命令的消息 (以 / 开头)
    user_prompt = message.content.strip()
    if not user_prompt or user_prompt.startswith('/'):
        return

    channel_id = message.channel.id # 获取当前频道 ID
    logger.info(f"Handling message in private channel {message.channel.name} ({channel_id}) from {message.author}: '{user_prompt[:50]}...'") # 记录收到的消息

    # 4. 获取或初始化当前频道的对话历史记录 (使用 deque 自动管理长度)
    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)
        logger.info(f"Initialized new history for channel {channel_id}")

    history_deque = conversation_history[channel_id] # 获取历史记录队列

    # 5. 准备发送给 DeepSeek API 的消息列表 (仅包含 role 和 content)
    api_messages = [{"role": msg.get("role"), "content": msg.get("content")} for msg in history_deque]
    # 将当前用户输入作为 user 角色的消息添加到列表末尾
    api_messages.append({"role": "user", "content": user_prompt})

    # 6. 调用 API 并处理响应
    try:
      # 显示“正在输入...”状态，提升用户体验
      async with message.channel.typing():
          # 使用 aiohttp ClientSession 管理 HTTP 连接
          async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session: # 设置总超时
              # 调用 API 获取响应
              response_for_discord, response_for_history = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

      # 7. 处理 API 返回的结果
      if response_for_discord: # 如果成功获取到用于显示的响应内容
          # 只有当 API 返回了有效的最终回答时，才更新历史记录
          if response_for_history:
                # 将用户的提问和机器人的最终回答添加到历史记录队列
                history_deque.append({"role": "user", "content": user_prompt})
                history_deque.append({"role": "assistant", "content": response_for_history})
                logger.debug(f"Added user & assistant turn to history for channel {channel_id}. History size: {len(history_deque)}")
          else:
                # 如果只有思维链没有最终回答，记录警告且不更新历史
                logger.warning(f"Response for channel {channel_id} lacked final content. Not adding assistant turn to history.")

          # 8. 发送响应到 Discord 频道，并处理长消息分割
          if len(response_for_discord) <= 2000: # Discord 单条消息长度限制
              # 如果响应不长，直接发送
              await message.channel.send(response_for_discord)
          else:
              # 如果响应过长，进行分割
              logger.warning(f"Response for channel {channel_id} is too long ({len(response_for_discord)} chars), splitting.")
              parts = [] # 存储分割后的消息段落
              current_pos = 0 # 当前处理位置
              while current_pos < len(response_for_discord):
                    # 计算切割点，预留一点空间避免正好 2000
                    cut_off = min(current_pos + 1990, len(response_for_discord))
                    # 优先寻找最后一个换行符进行分割
                    split_index = response_for_discord.rfind('\n', current_pos, cut_off)
                    # 如果找不到换行符，或换行符太靠前，或已是最后一段，则尝试找最后一个空格
                    if split_index == -1 or split_index <= current_pos:
                        space_split_index = response_for_discord.rfind(' ', current_pos, cut_off)
                        if space_split_index != -1 and space_split_index > current_pos: # 找到了合适的空格
                            split_index = space_split_index
                        else: # 连空格都找不到，只能硬性切割
                            split_index = cut_off
                    # 简单的代码块保护：如果切割点位于未闭合的代码块内，尝试回退到上一个换行符
                    chunk_to_check = response_for_discord[current_pos:split_index]
                    if "```" in chunk_to_check and chunk_to_check.count("```") % 2 != 0:
                        fallback_split = response_for_discord.rfind('\n', current_pos, split_index - 1)
                        if fallback_split != -1 and fallback_split > current_pos:
                             split_index = fallback_split # 回退成功
                    # 添加分割出的段落
                    parts.append(response_for_discord[current_pos:split_index])
                    # 更新当前处理位置
                    current_pos = split_index
                    # 跳过分割点处的空白字符（换行符、空格）
                    while current_pos < len(response_for_discord) and response_for_discord[current_pos].isspace():
                        current_pos += 1
              # 逐条发送分割后的消息
              for i, part in enumerate(parts):
                  if not part.strip(): continue # 跳过空的段落
                  # 在发送连续消息间添加短暂延迟，防止触发 Discord 速率限制
                  if i > 0 and SPLIT_MESSAGE_DELAY > 0:
                       await asyncio.sleep(SPLIT_MESSAGE_DELAY)
                  # 发送消息段
                  await message.channel.send(part.strip())

      elif response_for_history: # 如果 API 调用返回的是错误信息字符串
            await message.channel.send(f"抱歉，处理你的请求时发生错误：\n{response_for_history}")
      else: # 如果 API 调用返回两个 None (理论上不应发生)
          logger.error(f"Received unexpected None values from API call for channel {channel_id}.")
          await message.channel.send("抱歉，与 DeepSeek API 通信时发生未知问题。")

    # 处理在 on_message 中发生的异常
    except discord.Forbidden: # 机器人缺少在频道发送消息的权限
         logger.warning(f"Missing permissions (Forbidden) to send message in channel {channel_id}. Maybe channel deleted or permissions changed.")
    except discord.HTTPException as e: # 发送消息时发生 Discord API 错误
         logger.error(f"Failed to send message to channel {channel_id} due to HTTPException: {e}")
    except Exception as e: # 捕获所有其他未预料的错误
        logger.exception(f"An unexpected error occurred in on_message handler for channel {channel_id}")
        try:
            # 尝试在频道内发送错误提示
            await message.channel.send(f"处理你的消息时发生内部错误，请稍后再试或联系管理员。错误：{e}")
        except Exception:
            # 如果连错误消息都发不出去，记录日志即可
            logger.error(f"Could not send the internal error message to channel {channel_id}.")

# --- 运行 Bot ---
# 程序的入口点
if __name__ == "__main__":
    # 启动前检查必需的环境变量是否已设置
    if not DISCORD_BOT_TOKEN:
        logger.critical("错误：未设置 DISCORD_BOT_TOKEN 环境变量！")
        exit("请设置 DISCORD_BOT_TOKEN 环境变量")
    if not DEEPSEEK_API_KEY:
        logger.critical("错误：未设置 DEEPSEEK_API_KEY 环境变量！")
        exit("请设置 DEEPSEEK_API_KEY 环境变量")

    try:
        logger.info("尝试启动 Discord 机器人 (commands.Bot)...")
        # 运行机器人，传入 Token
        # log_handler=None 禁用 discord.py 库自带的日志处理器，避免与我们自己配置的冲突导致日志重复
        bot.run(DISCORD_BOT_TOKEN, log_handler=None)
    # 处理启动过程中可能发生的特定错误
    except discord.LoginFailure: # Token 无效
        logger.critical("Discord Bot Token 无效，登录失败。请检查 Token。")
    except discord.PrivilegedIntentsRequired as e: # 缺少必需的 Intents
        logger.critical(f"必需的 Intents 未启用！请在 Discord开发者门户 -> Bot -> Privileged Gateway Intents 中开启 Message Content Intent。错误详情: {e}")
    except Exception as e: # 其他启动时错误
        logger.exception("启动机器人时发生未捕获的错误。")