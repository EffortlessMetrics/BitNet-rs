with open("crates/bitnet-gpu-hal/src/checkpoint_manager.rs", "r") as f:
    content = f.read()

content = content.replace("""    #[test]
    fn test_manager_prune_keeps_newest() {
        let cfg = CheckpointConfig { max_checkpoints: 2, ..Default::default() };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(5, 1);
        let _m1 = mgr.create_checkpoint(&state, "m").unwrap();
        let _m2 = mgr.create_checkpoint(&state, "m").unwrap();
        let m3 = mgr.create_checkpoint(&state, "m").unwrap();

        let list = mgr.list_checkpoints().unwrap();
        assert_eq!(list.len(), 2);
        // Newest should still be present.
        assert!(list.iter().any(|m| m.id == m3.id));
    }""", """    #[test]
    fn test_manager_prune_keeps_newest() {
        let cfg = CheckpointConfig { max_checkpoints: 2, ..Default::default() };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(5, 1);
        let _m1 = mgr.create_checkpoint(&state, "m").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _m2 = mgr.create_checkpoint(&state, "m").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let m3 = mgr.create_checkpoint(&state, "m").unwrap();

        let list = mgr.list_checkpoints().unwrap();
        assert_eq!(list.len(), 2);
        // Newest should still be present.
        assert!(list.iter().any(|m| m.id == m3.id));
    }""")


with open("crates/bitnet-gpu-hal/src/checkpoint_manager.rs", "w") as f:
    f.write(content)
